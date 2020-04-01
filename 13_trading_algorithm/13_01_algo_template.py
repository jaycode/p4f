# ---------------------------------------------------
# 1. Imports and Settings
# ---------------------------------------------------



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


# ---------------------------------------------------
# END - 2. Helpers
# ---------------------------------------------------


# ---------------------------------------------------
# 3. Build Pipeline
# ---------------------------------------------------


# ---------------------------------------------------
# END - 3. Build Pipeline
# ---------------------------------------------------


# ---------------------------------------------------
# 6. Get Alpha Factors
# ---------------------------------------------------



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
