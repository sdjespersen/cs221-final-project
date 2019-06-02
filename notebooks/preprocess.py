import pandas as pd

INTERESTING_COLS = """
fire_size
fire_year
discovery_date
discovery_time
stat_cause_code
stat_cause_descr
cont_date
cont_doy
cont_time
latitude
longitude
state
county
fips_code
nwcg_reporting_unit_id
""".strip().split("\n")

FEATURE_COLS = """
fire_size
fire_year
discovery_date
burn_time
burn_time_notna
latitude
longitude
""".strip().split("\n")

def normalize(df):
  newdf = df.copy()
  # fire_year is between 1992 and 2015
  newdf.fire_year = (df.fire_year - 1992) / (2015 - 1992)
  # fire_size is between 0 and ~100,000 (acres)
  newdf.fire_size = 1. * df.fire_size / 100000
  # normalize these Julian dates
  mindd, maxdd = df.discovery_date.min(), df.discovery_date.max()
  newdf.discovery_date = (df.discovery_date - mindd) / (maxdd - mindd)
  # burn_time is between 0 and ~5,000
  newdf.burn_time = 1. * df.burn_time / 5000
  # shove -1 into burn_time where we don't have it;
  # the NN should pick up on this
  newdf.burn_time.fillna(-1, inplace=True)
  # latitude: take (-90, 90) -> (-1, 1)
  newdf.latitude /= 90
  # longitude: take (-180, 180) -> (-1, 1)
  newdf.longitude /= 180
  return newdf


def load_dataset(df_path):
  """Returns an (X, Y) pair of DataFrames given the data location."""
  df = pd.read_parquet(df_path)
  df.rename(columns={s: s.lower() for s in df.columns}, inplace=True)
  df.stat_cause_code = df.stat_cause_code.astype(int)
  df = df.filter(items=INTERESTING_COLS)
  # Drop codes 9 and 13, corresponding to miscellaneous and unknown causes,
  # which are pretty useless.
  df = df[(df.stat_cause_code != 9) & (df.stat_cause_code != 13)]
  label_df = pd.get_dummies(df.stat_cause_descr)
  df['burn_time'] = df.cont_date - df.discovery_date
  df['burn_time_notna'] = df.burn_time.notna().astype(int)
  data = df.filter(items=FEATURE_COLS)
  data = data.join(pd.get_dummies(df.state))
  data = data.rename(columns={s: s.lower() for s in data.columns})
  return normalize(data), label_df
