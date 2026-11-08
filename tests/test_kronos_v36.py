import unittest
import numpy as np
import pandas as pd
from scripts.kronos_v36_research import evaluate, summarize

class V36Tests(unittest.TestCase):
    def sample(self,n=180):
        dates=pd.bdate_range("2025-01-01",periods=n)
        return pd.DataFrame({"rank":range(n-1,-1,-1),
          "tw_origin":dates.astype(str),"session":"TW_CLOSE",
          "market":"TAIEX","horizon":1,
          "pred_return_pct":np.repeat(0.2,n),
          "realized_market_return_pct":np.repeat(0.3,n)})

    def test_output_and_baselines(self):
        out=evaluate(self.sample(),min_train=120)
        self.assertTrue({"raw","calibrated","always_up","always_down","trend_filtered"}.issubset(set(out.strategy)))
        s=summarize(out)
        up=s[s.strategy.eq("always_up")].iloc[0]
        self.assertEqual(up.direction_accuracy,1.0)

    def test_future_labels_do_not_change_past_signal(self):
        x=self.sample()
        before=evaluate(x,min_train=120)
        x.loc[x.index[-1],"realized_market_return_pct"]=-40.0
        after=evaluate(x,min_train=120)
        a=before[before.strategy.eq("calibrated")].reset_index(drop=True)
        b=after[after.strategy.eq("calibrated")].reset_index(drop=True)
        self.assertTrue((a.signal==b.signal).all())

    def test_duplicate_rejected(self):
        x=self.sample()
        with self.assertRaises(ValueError):
            evaluate(pd.concat([x,x.iloc[:1]]),min_train=120)

if __name__=="__main__":
    unittest.main()
