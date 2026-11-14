"""Verify evidence and leakage controls against the shipped raw observations."""
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.analysis import ROOT, SNAPSHOT_SHA256, TARGET, load_and_audit
from src.features import FUEL, NUMERIC, FuelInteractionFeatures


class ProjectEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.audit=load_and_audit(ROOT/'data/raw/my2025-fuel-consumption-ratings.csv')
        cls.metrics=json.loads((ROOT/'results/metrics.json').read_text())
        cls.pred=pd.read_csv(ROOT/'results/test_predictions.csv')

    def test_source_snapshot_and_audit(self):
        self.assertEqual(self.audit['sha256'],SNAPSHOT_SHA256)
        self.assertEqual(len(self.df),701)
        self.assertEqual(self.audit['exact_duplicates_removed'],0)
        self.assertFalse(self.df[NUMERIC+[TARGET]].isna().any().any())

    def test_catalog_groups_do_not_overlap(self):
        split=pd.read_csv(ROOT/'results/split_assignments.csv')
        train=set(split.loc[split.split=='train','group'])
        test=set(split.loc[split.split=='test','group'])
        self.assertFalse(train & test)
        self.assertEqual(len(train),480)
        self.assertEqual(len(test),120)
        self.assertEqual(len(split),len(self.df))

    def test_all_reported_test_metrics_match_predictions(self):
        for item in self.metrics['models']:
            p=self.pred[self.pred.model==item['name']]
            self.assertEqual(len(p),137)
            for key,fn in [('mae',mean_absolute_error),('mse',mean_squared_error),('r2',r2_score)]:
                self.assertAlmostEqual(item['test'][key],fn(p.actual,p.predicted),places=8)
            raw=self.df.set_index('source_row').loc[p.source_row,TARGET].to_numpy()
            np.testing.assert_allclose(raw,p.actual)

    def test_selection_uses_training_cv(self):
        cv=pd.read_csv(ROOT/'results/cross_validation.csv')
        means=cv.groupby('name').mae.mean()
        self.assertEqual(means.idxmin(),self.metrics['selected_model'])
        self.assertTrue((cv.unseen_fuel_rows==0).all())

    def test_serialized_model_reproduces_heldout_predictions(self):
        p=self.pred[self.pred.model==self.metrics['selected_model']]
        rows=self.df.set_index('source_row').loc[p.source_row,[*NUMERIC,FUEL]]
        model=joblib.load(ROOT/'models/selected_model.joblib')
        np.testing.assert_allclose(model.predict(rows),p.predicted,rtol=1e-10,atol=1e-8)
        self.assertEqual(set(model[0].get_feature_names_out()) & {TARGET,'CO2 rating','Smog rating'},set())

    def test_transformer_handles_one_fuel_and_unseen_code(self):
        x=pd.DataFrame({NUMERIC[0]:[2.,3.],NUMERIC[1]:[4,6],NUMERIC[2]:[7.,9.],FUEL:['X','X']})
        transformer=FuelInteractionFeatures().fit(x)
        unseen=x.copy(); unseen[FUEL]=['D','E']
        np.testing.assert_allclose(transformer.transform(unseen),x[NUMERIC])


if __name__=='__main__':
    unittest.main(verbosity=2)
