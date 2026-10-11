import numpy as np
import pandas as pd
import pytest
from pipeline import synthetic,preprocessing,prepare,partition,EMBARGO

def test_future_targets_do_not_change_training_preprocessing():
    d=synthetic();cut=d.timestamp.max().normalize()-pd.Timedelta(days=5)
    before=preprocessing(d,cut);changed=d.copy();changed.loc[changed.timestamp>=cut,'target_2h']=1e9
    assert preprocessing(changed,cut)==before

def test_partition_rejects_crossing_and_embargo_windows():
    cuts=[pd.Timestamp('2025-01-10'),pd.Timestamp('2025-01-12'),pd.Timestamp('2025-01-14')]
    assert partition(cuts[0]-pd.Timedelta(hours=1),cuts[0]+pd.Timedelta(hours=1),cuts) is None
    assert partition(cuts[0],cuts[0]+pd.Timedelta(hours=1),cuts) is None
    assert partition(cuts[0]+EMBARGO,cuts[1]-pd.Timedelta(minutes=30),cuts)=='validation'

def test_windows_are_causal_and_holdouts_nonoverlapping():
    _,_,_,_,audit,cuts=prepare(synthetic());d=pd.DataFrame(audit)
    for col in ['first_target','last_target','last_input']:d[col]=pd.to_datetime(d[col])
    assert (d.last_input<d.first_target).all()
    for _,g in d[d.split!='train'].groupby(['route','split']):
        g=g.sort_values('first_target');assert (g.first_target.iloc[1:].to_numpy()>g.last_target.iloc[:-1].to_numpy()).all()
    for left,right in [('train','validation'),('validation','calibration'),('calibration','test')]:
        assert d.loc[d.split==right,'first_target'].min()-EMBARGO>d.loc[d.split==left,'last_target'].max()

def test_duplicate_observation_rejected():
    d=synthetic()
    with pytest.raises(ValueError,match='Duplicate'):prepare(pd.concat([d,d.iloc[:1]]))
