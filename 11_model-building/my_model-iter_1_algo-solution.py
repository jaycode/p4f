# ---------------------------------------------------
# 1. Imports and Settings
# ---------------------------------------------------

# Module Imports
# --------------------
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import Fundamentals

from quantopian.pipeline.filters import QTradableStocksUS

import numpy as np
import pandas as pd
from datetime import datetime

# Environment Settings
# --------------------
## Production 
universe = QTradableStocksUS()
mask = {'mask': universe}

## Development
# universe = QTradableStocksUS()
# mask = {'mask': universe}


# Global Configuration
# --------------------


# ---------------------------------------------------
# END - 1. Imports and Settings
# ---------------------------------------------------


# ---------------------------------------------------
# Algo Imports and Parameters
# ---------------------------------------------------
import quantopian.algorithm as algo
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.experimental import risk_loading_pipeline

NO_COST = False   # disable trading costs and slippage

# Constraints-related configurations:
# MAX_GROSS_LEVERAGE = 1.0
# MAX_SHORT_POSITION_SIZE = 0.01  # 1%
# MAX_LONG_POSITION_SIZE = 0.01   # 1%

# ---------------------------------------------------
# END - Algo Imports and Parameters
# ---------------------------------------------------


# ---------------------------------------------------
# 2. Helpers
# ---------------------------------------------------

def normalize(data):
    """ Normalize long/short positions
    """
    result = pd.Series(data)
    result = result - np.nanmean(data)
        
    denom = result.abs().sum()
    if denom == 0:
        denom = 1
    
    return result / denom


class Normalize(CustomFactor):
    window_length = 1
    def compute(self, today, assets, out, inp):
        data = normalize(inp[-1, :])
        out[:] = data

# ---------------------------------------------------
# END - 2. Helpers
# ---------------------------------------------------


# ---------------------------------------------------
# 3. Build Pipeline
# ---------------------------------------------------

def make_alpha_factors():
    factors = {}
    # Create factors here
    a_pe_ratio = -Fundamentals.pe_ratio.latest.zscore()
    factors['a_pe_ratio'] = a_pe_ratio
    
    a_pb_ratio = -Fundamentals.pb_ratio.latest.zscore()
    factors['a_pb_ratio'] = a_pb_ratio
    
    factor_sum = a_pe_ratio + a_pb_ratio
    factors['a_combined'] = Normalize(inputs=[factor_sum], **mask)
    
    # Not an alpha factor, but useful for later in the grouped tear sheet analysis
    factors['sector'] = Fundamentals.morningstar_sector_code.latest
    return factors
                                        

def make_pipeline():
    alpha_factors = make_alpha_factors()
    factors = {a: alpha_factors[a] for a in alpha_factors}
    pipe = Pipeline(columns=factors, screen=universe)
    
    return pipe

# ---------------------------------------------------
# END - 3. Build Pipeline
# ---------------------------------------------------


# ---------------------------------------------------
# 6. Get Alpha Factors
# ---------------------------------------------------

def get_alpha(factors):
    # Replace infs and NaNs
    factors[np.isinf(factors)] = np.nan
    factors.fillna(0, inplace=True)

    combined_alpha = factors['a_combined']
    return combined_alpha

# ---------------------------------------------------
# END - 6. Get Alpha Factors
# --------------------------------------------------


def initialize(context):
    # Rebalance every day, after market close
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(hours=1)
    )

    if NO_COST:
        set_commission(commission.PerShare(cost=0, min_trade_cost=0))
        set_slippage(slippage.FixedBasisPointsSlippage(basis_points=0, volume_limit=0.1))

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')


def before_trading_start(context, data):
    # Get the risk loading data every day.
    context.risk_loading_pipeline = pipeline_output('risk_loading_pipeline')


def rebalance(context, data):
    # Get today's alphas as a MultiIndex DataFrame
    mdf = (pipeline_output('pipeline')).astype('float64')

    # Combine the alpha factors
    combined_alpha = get_alpha(mdf)
    
    # Define the objective
    objective = opt.TargetWeights(combined_alpha)


# ---------------------------------------------------
# Risk Constraints
# ---------------------------------------------------

# Uncomment the code below to add constraints
#

#     # Define the position concentration constraint.
#     constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
#         -MAX_SHORT_POSITION_SIZE,
#         MAX_LONG_POSITION_SIZE,
#     )

#     # Constrain our risk exposures. We're using version 0 of the default bounds
#     # which constrain our portfolio to 18% exposure to each sector and 36% to
#     # each style factor.
#     constrain_sector_style_risk = opt.experimental.RiskModelExposure(  
#         risk_model_loadings=context.risk_loading_pipeline,  
#         version=0,
#     )
    
#     # Define the max leverage constraint.
#     constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
#     # Define the dollar neutral constraint.
#     dollar_neutral = opt.DollarNeutral()
# 
    # Add our constraints, comment or uncomment as necessary.
    constraints = [
        # constrain_sector_style_risk,
        # constrain_gross_leverage,
        # dollar_neutral,
        # constrain_pos_size
    ]

# ---------------------------------------------------
# END - Risk Constraints
# ---------------------------------------------------
    
    # Calculate the optimal portfolio
    try:
        combined_alpha = opt.calculate_optimal_portfolio(objective=objective, constraints=constraints)
    except:
        pass

    # Drop expired securites (i.e. that aren't in the tradeable universe on that date)
    combined_alpha = combined_alpha[combined_alpha.index.isin(pipeline_output('pipeline').index)]

    # Do a final null filter and normalization
    combined_alpha = combined_alpha[pd.notnull(combined_alpha)]
    combined_alpha = normalize(combined_alpha)

    # Define the objective
    objective = opt.TargetWeights(combined_alpha)

    # Order the optimal portfolio
    try:
        order_optimal_portfolio(objective=objective, constraints=constraints)
    except:
        pass


def record_vars(context, data):
    record(num_positions=len(context.portfolio.positions))
