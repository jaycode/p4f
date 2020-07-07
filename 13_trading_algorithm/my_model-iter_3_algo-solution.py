# ---------------------------------------------------
# 1. Imports and Settings
# ---------------------------------------------------
# Module Imports
# --------------------
# Module Imports
# --------------------
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import Fundamentals

from quantopian.pipeline.filters import QTradableStocksUS, StaticAssets

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import gmean

# Environment Settings
# --------------------
## Production 
universe = QTradableStocksUS()
mask = {'mask': universe}

## Development
# universe = StaticAssets(symbols(['FB']))
# mask = {'mask': universe}


# Global Configuration
# --------------------

# None, 'industry', 'sector'
SCALE_BY = 'sector'
PIPE_NORMALIZE = True
CLIP_OUTLIERS = False
CLIP_THRESHOLD =  0.025

# When true, for each day, get some % of shorts and some % of longs
USE_EXTREMES = False
EXTREMES_BOTTOM = 0.03
EXTREMES_TOP = 0.07
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
def get_annual(data, t=251, num_years=None):
    data = data[::t,:]
    if num_years != None:
        data = data[-num_years:]
    return data


def get_changes(data, t=1):
    data = pd.DataFrame(data)
    prev_data = data.shift(t, axis=0)
    return data/prev_data


def zscore(data):
    result = pd.Series(data)
    result = result - np.nanmean(data)
        
    denom = result.std()

    return result / denom


def standardize(data, standardize_by=None, standardizer=None):
    data = pd.Series(data)
    
    # Prepare the data
    dfData = pd.DataFrame({'data': data})
    if standardize_by != None and standardizer is not None:
        dfData[standardize_by] = standardizer
    
        # Standardize the data
        zscore = lambda x: (x - x.mean()) / (x.std() == 0 and 1 or x.std())
        data = dfData.groupby([standardize_by])['data'].transform(zscore)
    
    return data


def normalize(data):
    """ Normalize long/short positions
    """
    result = pd.Series(data)
    result = result - np.nanmean(data)
        
    denom = result.abs().sum()
    if denom == 0:
        denom = 1
    
    return result / denom


def clip(data, threshold=0.025, drop=False):
    data = pd.Series(data)
    data_notnull = data[data.notnull()]
    if data_notnull.shape[0] > 0:
        low_cutoff = data_notnull.quantile(threshold)
        high_cutoff = data_notnull.quantile(1 - threshold)
        if not drop:
            data = data.clip(lower=low_cutoff, upper=high_cutoff).values
        else:
            data = data[(data < low_cutoff) | (data > high_cutoff)]
    
    return data


def get_extremes(data, n_bottom_or_both, n_top=None):
    """ Divide data into 100 bins, then get n top and bottom bins.
    """
    data_notnan = data[~np.isnan(data)]
    if n_top != None:
        n_bottom = n_bottom_or_both
        condition = (data < np.percentile(data_notnan, float(n_bottom))) | \
                    (data > np.percentile(data_notnan, 1.0-n_top))
    else:
        n = n_bottom_or_both
        condition = (data < np.percentile(data_notnan, float(n))) | \
                    (data > np.percentile(data_notnan, 1.0-n))
    return np.where(condition, data, np.nan)


class Scale(CustomFactor):
    """ Scale by either industry or sector.
    This is for a quick experiment to get scaled value
    of the last day. Once multiple dates are needed,
    a new factor needs to be created.
    """
    # inputs = [factor, classifier]
    inputs = [USEquityPricing.close.latest,
              Fundamentals.morningstar_industry_code.latest]
    window_length = 1
    params={
        'clip_outliers': False,
        'clip_threshold': 0.025,
        'scale_by': 'sector', # None, 'sector' or 'industry'
    }
    
    def compute(self, today, assets, out, input1, groupby, clip_outliers,
                clip_threshold, scale_by):
        data = input1[-1, :]

        out[:] = self.transform(data, groupby, clip_outliers,
                                clip_threshold, scale_by)
    
    
    def transform(self, data, groupby, clip_outliers,
                  clip_threshold, scale_by):
        
        if clip_outliers:
            data = clip(data, threshold=clip_threshold)

        groupby_data = groupby[-1, :]
        data = standardize(data,
                           standardize_by=scale_by,
                           standardizer=groupby_data)
        return data

    
class AnnualScale(Scale):
    """ Scale fundamental data.
    Use annual instead of daily values."""
    
    # asof_dates, values, groupby
    inputs = [Fundamentals.roa_asof_date, Fundamentals.roa,
              Fundamentals.morningstar_industry_code]
    
    params = {
        **{
            'num_years': 1
        },
        **Scale.params}


class CFOA(AnnualScale):
    inputs = [Fundamentals.free_cash_flow_asof_date,
              Fundamentals.free_cash_flow, Fundamentals.total_assets,
              Fundamentals.morningstar_sector_code]
    def compute(self, today, assets, out, asof_date, fcf, ta, groupby,
                num_years,
                clip_outliers, clip_threshold, scale_by):
        fcf_ann = get_annual(fcf, num_years=num_years)
        fcf_sum = np.sum(fcf_ann, axis=0)
        ta = ta[-1, :]
        ta[ta == 0] = 1
        result = fcf_sum/ta
        
        out[:] = self.transform(result, groupby, clip_outliers,
                                clip_threshold, scale_by)
        
        
class GMean(AnnualScale):
    inputs = [Fundamentals.roa_asof_date,
              Fundamentals.roa,
              Fundamentals.morningstar_sector_code]
    def compute(self, today, assets, out, asof_date, inp, groupby,
                num_years,
                clip_outliers, clip_threshold, scale_by):
        inp_ann = get_annual(inp, num_years=num_years)
        inp_ann[np.isnan(inp_ann)] = 0.0
        result = gmean(inp_ann+1, axis=0)-1

        out[:] = self.transform(result, groupby, clip_outliers,
                                clip_threshold, scale_by)        


class GMeanChanges(AnnualScale):
    inputs = [Fundamentals.roa_asof_date,
              Fundamentals.roa,
              Fundamentals.morningstar_industry_code]
    def compute(self, today, assets, out, asof_date, inp, groupby,
                num_years,
                clip_outliers, clip_threshold, scale_by):
        inp_ann = get_annual(inp, num_years=num_years)
        changes = get_changes(inp_ann)[1:]
        changes[np.isnan(changes)] = 0.0
        result = gmean(changes+1, axis=0)-1

        out[:] = self.transform(result, groupby, clip_outliers,
                                clip_threshold, scale_by)
        
        
class MeanOverStd(AnnualScale):
    def compute(self, today, assets, out, asof_date, value, groupby,
                num_years,
                clip_outliers, clip_threshold, scale_by):
        value_ann = get_annual(value, num_years=num_years)
        result = np.nanmean(value_ann, axis=0) / np.nanstd(value_ann, axis=0)        
        result[result == np.inf] = np.nan
        result[result == -np.inf] = np.nan

        out[:] = self.transform(result, groupby, clip_outliers,
                                        clip_threshold, scale_by)
        
        
class Max(CustomFactor):
    inputs = [],
    window_length = 1
    def compute(self, today, assets, out, inp1, inp2):
        combined = np.dstack([inp1[-1, :], inp2[-1, :]])
        result = np.amax(combined, axis=2)
        out[:] = result
        

# --- For combined factors ---

class Normalize(CustomFactor):
    window_length = 1
    def compute(self, today, assets, out, inp):
        data = normalize(inp[-1, :])
        out[:] = data

        
class Extremes(CustomFactor):
    """ Get both extremes of a factor.
    
    Pass in percentiles from the bottom and top.
    If normalized is True, normalize by taking into account
    long and short trades (use to create the final alpha factor).
    """
    window_length = 1
    params = {
        'extremes_bottom': 0.03,
        'extremes_top': 0.07,
        'normalized': False
    }
    def compute(self, today, assets, out, inp,
                extremes_bottom, extremes_top, normalized):
        data = inp[-1, :]
        data = get_extremes(data, extremes_bottom, extremes_top)
        if normalized:
            data = normalize(data)
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

    if SCALE_BY == 'industry':
        scaler = Fundamentals.morningstar_industry_code
    elif SCALE_BY == 'sector':
        scaler = Fundamentals.morningstar_sector_code

    # gs = group-standardized
    a_pe_ratio_gs = -Scale(
        inputs=[Fundamentals.pe_ratio.latest, scaler.latest],
        window_length=1,
        scale_by=SCALE_BY,
        clip_outliers=CLIP_OUTLIERS,
        clip_threshold=CLIP_THRESHOLD,
        **mask)
    factors['a_pe_ratio_gs'] = a_pe_ratio_gs

    a_pb_ratio_gs = -Scale(
        inputs=[Fundamentals.pb_ratio.latest, scaler.latest],
        window_length=1,
        scale_by=SCALE_BY,
        clip_outliers=CLIP_OUTLIERS,
        clip_threshold=CLIP_THRESHOLD,
        **mask)
    factors['a_pb_ratio_gs'] = a_pb_ratio_gs
    
    a_roic = Scale(
        inputs=[Fundamentals.roic.latest, scaler.latest],
        window_length=1,
        scale_by=SCALE_BY,
        clip_outliers=CLIP_OUTLIERS,
        clip_threshold=CLIP_THRESHOLD,
        **mask)
    factors['a_roic'] = a_roic
    
    # 2. CFOA
    a_cfoa = CFOA(inputs=[Fundamentals.free_cash_flow_asof_date,
                          Fundamentals.free_cash_flow,
                          Fundamentals.total_assets,
                          scaler],
                  window_length=252*8,
                  num_years=8,
                  clip_outliers=CLIP_OUTLIERS,
                  clip_threshold=CLIP_THRESHOLD,
                  scale_by=SCALE_BY,
                  **mask)
    factors['a_cfoa'] = a_cfoa
    
    # 3. Persistent Return
    a_8yr_roa = GMean(inputs=[Fundamentals.roa_asof_date,
                              Fundamentals.roa,
                              scaler],
                      window_length=252*8,
                      num_years=8,
                      clip_outliers=CLIP_OUTLIERS,
                      clip_threshold=CLIP_THRESHOLD,
                      scale_by=SCALE_BY,
                      **mask)
    factors['a_8yr_roa'] = a_8yr_roa
    
    a_8yr_roic = GMean(inputs=[Fundamentals.roic_asof_date,
                               Fundamentals.roic,
                               scaler],
                       window_length=252*8,
                       num_years=8,
                       clip_outliers=CLIP_OUTLIERS,
                       clip_threshold=CLIP_THRESHOLD,
                       scale_by=SCALE_BY,
                       **mask)
    factors['a_8yr_roic'] = a_8yr_roic
    
    # 4 Maximum Margin
    mg = GMeanChanges(inputs=[Fundamentals.gross_margin_asof_date,
                             Fundamentals.gross_margin,
                             scaler],
               window_length=252*8,
               num_years=8,
               clip_outliers=CLIP_OUTLIERS,
               clip_threshold=CLIP_THRESHOLD,
               scale_by=SCALE_BY,
               **mask)
    ms = MeanOverStd(inputs=[Fundamentals.gross_margin_asof_date,
                             Fundamentals.gross_margin,
                             scaler],
                     window_length=252*8,
                     num_years=8,
                     clip_outliers=CLIP_OUTLIERS,
                     clip_threshold=CLIP_THRESHOLD,
                     scale_by=SCALE_BY,
                     **mask)
    a_mm = Max(inputs=[mg, ms])
    factors['a_mm'] = a_mm
    
    factor_sum = a_pe_ratio_gs + a_pb_ratio_gs + a_roic + \
                 a_cfoa + a_8yr_roa + a_8yr_roic + \
                 a_mm
    if USE_EXTREMES:
        combined_alpha = Extremes(
                         inputs=[factor_sum],
                         extremes_bottom=EXTREMES_BOTTOM,
                         extremes_top=EXTREMES_TOP,
                         normalized=PIPE_NORMALIZE,
        )
    else:
        if PIPE_NORMALIZE:
            combined_alpha = Normalize(
                             inputs=[factor_sum],
            )
        else:
            combined_alpha = factor_sum
    factors['a_combined'] = combined_alpha

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
    