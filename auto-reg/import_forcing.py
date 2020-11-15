import pandas as pd
def import_forcing(address):
    forcingT = pd.read_feather(address)
    site_no = forcingT['site_no'].unique()

    forcingT.loc[(forcingT['00060_Mean'] != forcingT['00060_Mean']) & \
                 (forcingT['datetime'] >= '2010-10-01') & \
                 (forcingT['datetime'] < '2016-10-01'), '00060_Mean'] = forcingT.loc[
        (forcingT['00060_Mean'] != forcingT['00060_Mean']) & \
        (forcingT['datetime'] >= '2010-10-01') & \
        (forcingT['datetime'] < '2016-10-01'), 'combine_discharge']

    forcing = forcingT.loc[(forcingT['datetime'] >= '2010-10-01') &
                           (forcingT['datetime'] < '2016-10-01')]

    col1 = ['prcp(mm/day)', 'tmax(C)',
            'tmin(C)', '00060_Mean', '00010_Mean', 'datetime', 'site_no']
    col = ['dayl(s)', 'srad(W/m2)', 'swe(mm)', 'vp(Pa)', 'site_no', '00010_Maximum',
           '00010_Mean', '00010_Minimum', 'site_no_from_TS',
           'pred_discharge', 'combine_discharge', 'datetime']

    f_s_0 = forcingT[col1]

    f_s_1 = forcingT.shift(periods=1)
    f_s_1 = f_s_1.drop(col, axis=1)

    f_s_2 = forcingT.shift(periods=2)
    f_s_2 = f_s_2.drop(col, axis=1)

    f_s_3 = forcingT.shift(periods=3)
    f_s_3 = f_s_3.drop(col, axis=1)

    f_s_4 = forcingT.shift(periods=4)
    f_s_4 = f_s_4.drop(col, axis=1)

    f_s_5 = forcingT.shift(periods=5)
    f_s_5 = f_s_5.drop(col, axis=1)

    ft = pd.concat([f_s_0, f_s_1, f_s_2, f_s_3, f_s_4, f_s_5], axis=1)

    ft.columns = ['prcp(mm/day)', 'tmax(C)', 'tmin(C)', '00060_Mean', '00010_Mean', 'datetime', 'site_no',
                  'prcp(mm/day)_1', 'tmax(C)_1', 'tmin(C)_1', '00060_Mean_1',
                  'prcp(mm/day)_2', 'tmax(C)_2', 'tmin(C)_2', '00060_Mean_2',
                  'prcp(mm/day)_3', 'tmax(C)_3', 'tmin(C)_3', '00060_Mean_3',
                  'prcp(mm/day)_4', 'tmax(C)_4', 'tmin(C)_4', '00060_Mean_4',
                  'prcp(mm/day)_5', 'tmax(C)_5', 'tmin(C)_5', '00060_Mean_5']

    forcing = ft.loc[(ft['datetime'] >= '2010-10-01') &
                     (ft['datetime'] < '2016-10-01')]
    return forcing, site_no