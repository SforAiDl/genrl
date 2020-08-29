from genrl.utils import (
    AdultDataBandit,
    CensusDataBandit,
    CovertypeDataBandit,
    MagicDataBandit,
    MushroomDataBandit,
    StatlogDataBandit,
)

from .utils import write_data


class TestDataBandit:
    def _test_fn(self, bandit) -> None:
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_covertype_data_bandit(self) -> None:
        d = """2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
2590,56,2,212,-6,390,220,235,151,6225,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
2804,139,9,268,65,3180,234,238,135,6121,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2
2785,155,18,242,118,3090,238,238,122,6211,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2
2595,45,2,153,-1,391,220,234,150,6172,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
"""
        fpath = write_data("covtype.data", d)
        bandit = CovertypeDataBandit(path=fpath)
        self._test_fn(bandit)
        fpath.unlink()

    def test_mushroom_data_bandit(self) -> None:
        d = """p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g
e,b,s,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,n,m
p,x,y,w,t,p,f,c,n,n,e,e,s,s,w,w,p,w,o,p,k,s,u
e,x,s,g,f,n,f,w,b,k,t,e,s,s,w,w,p,w,o,e,n,a,g
"""
        fpath = write_data("agaricus-lepiota.data", d)
        bandit = MushroomDataBandit(path=fpath)
        self._test_fn(bandit)
        fpath.unlink()

    def test_statlog_data_bandit(self) -> None:
        d = """50 21 77 0 28 0 27 48 22 2
55 0 92 0 0 26 36 92 56 4
53 0 82 0 52 -5 29 30 2 1
37 0 76 0 28 18 40 48 8 1
37 0 79 0 34 -26 43 46 2 1
"""
        fpath = write_data("shuttle.trn", d)
        bandit = StatlogDataBandit(path=fpath)
        self._test_fn(bandit)
        fpath.unlink()

    def test_adult_data_bandit(self) -> None:
        d = """31, Private, 84154, Some-college, 10, Married-civ-spouse, Sales, Husband, White, Male, 0, 0, 38, ?, >50K
18, Private, 226956, HS-grad, 9, Never-married, Other-service, Own-child, White, Female, 0, 0, 30, ?, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, <=50K
"""
        fpath = write_data("adult.data", d)
        bandit = AdultDataBandit(path=fpath)
        self._test_fn(bandit)
        fpath.unlink()

    def test_census_data_bandit(self) -> None:
        d = """caseid,dAge,dAncstry1,dAncstry2,iAvail,iCitizen,iClass,dDepart,iDisabl1,iDisabl2,iEnglish,iFeb55,iFertil,dHispanic,dHour89,dHours,iImmigr,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,iKorean,iLang1,iLooking,iMarital,iMay75880,iMeans,iMilitary,iMobility,iMobillim,dOccup,iOthrserv,iPerscare,dPOB,dPoverty,dPwgt1,iRagechld,dRearning,iRelat1,iRelat2,iRemplpar,iRiders,iRlabor,iRownchld,dRpincome,iRPOB,iRrelchld,iRspouse,iRvetserv,iSchool,iSept80,iSex,iSubfam1,iSubfam2,iTmpabsnt,dTravtime,iVietnam,dWeek89,iWork89,iWorklwk,iWWII,iYearsch,iYearwrk,dYrsserv
10000,5,0,1,0,0,5,3,2,2,1,0,1,0,4,3,0,2,0,0,1,0,0,0,0,10,0,1,0,1,0,1,4,2,2,3,0,2,0,2,1,4,3,0,0,0,3,1,0,3,22,0,3,0,1,0,1,0,0,0,5,0,2,1,1,0,11,1,0
10001,6,1,1,0,0,7,5,2,2,0,0,3,0,1,1,0,1,0,0,0,0,1,0,0,4,0,2,0,0,0,1,4,1,2,2,0,2,0,2,2,4,2,1,0,0,1,1,0,2,10,0,1,0,1,0,1,0,0,0,1,0,2,1,1,0,5,1,0
10002,3,1,2,0,0,7,4,2,2,0,0,1,0,4,4,0,1,0,1,0,0,0,0,0,1,0,2,0,4,0,10,4,1,2,4,0,2,0,2,1,4,2,2,0,0,0,1,0,2,10,0,6,0,1,0,1,0,0,0,2,0,2,1,1,0,10,1,0
10003,4,1,2,0,0,1,3,2,2,0,0,3,0,3,3,0,1,0,0,0,0,0,0,1,4,0,2,0,2,0,1,4,1,2,2,0,2,0,2,1,2,2,0,0,0,1,1,0,2,10,0,4,0,1,0,1,0,0,0,1,0,1,1,1,0,10,1,0
10004,7,1,1,0,0,0,0,2,2,0,0,3,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,2,0,0,0,4,1,2,0,0,2,0,2,1,4,0,1,0,0,0,6,0,2,22,0,1,0,1,0,1,0,0,3,0,0,0,2,2,0,5,6,0
"""
        fpath = write_data("USCensus1990.data.txt", d)
        bandit = CensusDataBandit(path=fpath)
        self._test_fn(bandit)
        fpath.unlink()

    def test_magic_data_bandit(self) -> None:
        d = """28.7967,16.0021,2.6449,0.3918,0.1982,27.7004,22.011,-8.2027,40.092,81.8828,g
31.6036,11.7235,2.5185,0.5303,0.3773,26.2722,23.8238,-9.9574,6.3609,205.261,g
162.052,136.031,4.0612,0.0374,0.0187,116.741,-64.858,-45.216,76.96,256.788,g
23.8172,9.5728,2.3385,0.6147,0.3922,27.2107,-6.4633,-7.1513,10.449,116.737,g
75.1362,30.9205,3.1611,0.3168,0.1832,-5.5277,28.5525,21.8393,4.648,356.462,g
"""
        fpath = write_data("magic04.data", d)
        bandit = MagicDataBandit(path=fpath)
        self._test_fn(bandit)
        fpath.unlink()
