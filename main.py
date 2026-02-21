import sys

from src.Classifiers.K2.Pipeline.K2CampaignRunner import K2CampaignRunner
from src.Classifiers.K2.Pipeline.K2FailedDownloader import K2FailedDownloader
from src.Classifiers.K2.Pipeline.K2PosthocRanking import K2PosthocRanking
from src.Classifiers.K2.Pipeline.K2PosthocRetriage import K2PosthocRetriage
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.CnnModel import CnnModel
from src.Classifiers.LargeWindow.LargeWindow_Processor import LargeWindowCnnModel


def main():
    argv = sys.argv[1:]
    if len(argv) > 0 and argv[0] == "k2_retry_failed":
        out = K2FailedDownloader.run_cli(argv=argv[1:])
        print(f"failed_recovery.csv: {out['failed_recovery_csv']}")
        print(f"progress_failed.json: {out['progress_failed_json']}")
        print(f"processed: {out['processed']}")
        return
    if len(argv) > 0 and argv[0] == "k2_retriage":
        out = K2PosthocRetriage.run_cli(argv=argv[1:])
        print(f"input: {out['input_csv']}")
        print(f"output: {out['out_csv']}")
        print(f"triage_status_ok_rows: {out['triage_ok_rows']}")
        print(f"relabeled_rows: {out['relabeled_rows']}")
        return
    if len(argv) > 0 and argv[0] == "k2_rank_posthoc":
        out = K2PosthocRanking.run_cli(argv=argv[1:])
        print(f"input: {out['input_csv']}")
        print(f"shortlist_top_shape.csv: {out['shortlist_top_shape_csv']}")
        print(f"shortlist_top_shape_for_period.csv: {out['shortlist_top_shape_for_period_csv']}")
        print(f"rows_shortlist_top_shape: {out['rows_shortlist_top_shape']}")
        print(f"rows_shortlist_top_shape_for_period: {out['rows_shortlist_top_shape_for_period']}")
        return
    if len(argv) > 0 and argv[0] == "k2_shortlist_period":
        out = K2ShortlistPeriodRunner().run()
        print(f"shortlist: {out['shortlist_csv']}")
        print(f"period_shortlist_summary.csv: {out['out_summary_csv']}")
        print(f"period_shortlist_summary_unique_epicP.csv: {out['out_summary_unique_epicp_csv']}")
        print(f"period_shortlist_summary_validated_only.csv: {out['out_summary_validated_only_csv']}")
        print(f"period_shortlist_best.csv: {out['out_best_csv']}")
        print(f"period_shortlist_quarantine.csv: {out['out_quarantine_csv']}")
        print(f"period_shortlist_diagnostics.csv: {out['out_diagnostics_csv']}")
        print(f"period_hist_summary_vs_best.png: {out['out_period_hist_png']}")
        print(f"period_hist_summary_vs_best_counts.csv: {out['out_period_hist_counts_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"rows_null_p: {out['rows_null_p']}")
        print(f"rows_dropped: {out['rows_dropped']}")
        print(f"rows_valid: {out['rows_valid']}")
        print(f"targets: {out['n_targets']}")
        print(f"best_rows: {out['n_best_rows']}")
        return
    K2CampaignRunner().run()


def largeWindowMain():
    largeWindowModel = LargeWindowCnnModel()
    #largeWindowModel.build_model(mission="tess",neg_pos_ratio= 3, do_hard_neg=True)
   # largeWindowModel.build_model(mission="kepler",neg_pos_ratio= 7 , do_hard_neg=False)
    largeWindowModel.build_model(mission="k2", neg_pos_ratio=2,do_hard_neg=False)

if __name__ == "__main__":
    main()
