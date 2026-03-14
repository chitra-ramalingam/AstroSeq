import sys

from src.Classifiers.K2.Pipeline.K2CampaignRunner import K2CampaignRunner
from src.Classifiers.K2.Pipeline.K2FailedDownloader import K2FailedDownloader
from src.Classifiers.K2.Pipeline.K2PosthocRanking import K2PosthocRanking
from src.Classifiers.K2.Pipeline.K2PosthocRetriage import K2PosthocRetriage
from src.Classifiers.K2.Pipeline.K2WhitenessRunner import K2WhitenessRunner
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2ShortlistPeriodCompare import K2ShortlistPeriodCompare
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
    if len(argv) > 0 and argv[0] == "k2_whiteness":
        out = K2WhitenessRunner.run_cli(argv=argv[1:])
        def _fmt_counts(d):
            if not isinstance(d, dict) or len(d) == 0:
                return "none"
            return " | ".join([f"{k}:{v}" for k, v in d.items()])

        print(f"rows_total: {out['total_rows']}")
        print(f"usable_count: {out['usable_rows']}")
        print(f"null_ok_rows_count: {out.get('null_ok_rows_count', 0)}")
        print(f"null_ok_rows_csv: {out.get('null_ok_rows_csv', '')}")
        print(
            "whiteness_quantiles: "
            f"min={out['whiteness_min']} median={out['whiteness_median']} max={out['whiteness_max']}"
        )
        print(f"whiteness_null_count: {out['whiteness_null_count']}")
        print(
            f"whiteness_near_zero_count(eps={out['whiteness_near_zero_eps']}): "
            f"{out['whiteness_near_zero_count']}"
        )
        print(f"whiteness_gt_0_95_count: {out['whiteness_gt_0_95_count']}")
        print(f"whiteness_lt_0_05_count: {out['whiteness_lt_0_05_count']}")
        print(f"whiteness_null_semantics: {out['whiteness_null_semantics']}")
        print(
            "null_whiteness_by_triage_status: "
            f"{_fmt_counts(out.get('null_whiteness_by_triage_status', {}))}"
        )
        print(
            "null_whiteness_by_triage_usable: "
            f"{_fmt_counts(out.get('null_whiteness_by_triage_usable', {}))}"
        )
        print(
            "null_whiteness_by_triage_why_not_usable_top: "
            f"{_fmt_counts(out.get('null_whiteness_by_triage_why_not_usable_top', {}))}"
        )
        print(f"null_whiteness_usable_true_count: {out.get('null_whiteness_usable_true_count', 0)}")
        print(f"null_whiteness_shortlist_attempt_count: {out.get('null_whiteness_shortlist_attempt_count', 0)}")
        print(f"null_whiteness_shortlist_topk_candidate_count: {out.get('null_whiteness_shortlist_topk_candidate_count', 0)}")
        deciles = out.get("whiteness_deciles", {})
        if isinstance(deciles, dict) and len(deciles) > 0:
            deciles_text = " ".join([f"{k}={v}" for k, v in deciles.items()])
            print(f"whiteness_deciles: {deciles_text}")
        hist_bins = out.get("whiteness_histogram_bins", {})
        if isinstance(hist_bins, dict) and len(hist_bins) > 0:
            hist_text = " | ".join([f"{k}:{v}" for k, v in hist_bins.items()])
            print(f"whiteness_histogram_10bins: {hist_text}")
            print(f"whiteness_outside_0_1_count: {out['whiteness_outside_0_1_count']}")
        bucket_table = out.get("whiteness_bucket_table", {})
        if isinstance(bucket_table, dict):
            print("whiteness_bucket_table:")
            for row in bucket_table.get("buckets", []):
                b = str(row.get("bucket", ""))
                c = int(row.get("count", 0))
                pct = float(row.get("pct_total", 0.0))
                print(f"  {b:10s} count={c:7d} pct_total={pct:6.2f}%")
            if "outside_0_1_count" in bucket_table:
                print(f"  outside_0_1 count={int(bucket_table.get('outside_0_1_count', 0)):7d}")
        return
    if len(argv) > 0 and argv[0] == "k2_rank_posthoc":
        out = K2PosthocRanking.run_cli(argv=argv[1:])
        print(f"input: {out['input_csv']}")
        print(f"shortlist_top_shape.csv: {out['shortlist_top_shape_csv']}")
        print(f"shortlist_top_shape_for_period.csv: {out['shortlist_top_shape_for_period_csv']}")
        print(f"rows_shortlist_top_shape: {out['rows_shortlist_top_shape']}")
        print(f"rows_shortlist_top_shape_for_period: {out['rows_shortlist_top_shape_for_period']}")
        print(f"period_stage_max_epics: {out['period_stage_max_epics']}")
        return
    if len(argv) > 0 and argv[0] == "k2_shortlist_period":
        out = K2ShortlistPeriodRunner.run_cli(argv=argv[1:])
        print(f"shortlist: {out['shortlist_csv']}")
        print(f"period_shortlist_summary.csv: {out['out_summary_csv']}")
        print(f"period_shortlist_summary_unique_epicP.csv: {out['out_summary_unique_epicp_csv']}")
        print(f"period_shortlist_summary_validated_only.csv: {out['out_summary_validated_only_csv']}")
        print(f"period_shortlist_best.csv: {out['out_best_csv']}")
        print(f"period_shortlist_quarantine.csv: {out['out_quarantine_csv']}")
        print(f"epic_funnel_reasons.csv: {out['out_epic_funnel_reasons_csv']}")
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
    if len(argv) > 0 and argv[0] == "k2_shortlist_period_compare":
        out = K2ShortlistPeriodCompare.run_cli(argv=argv[1:])
        print(f"rescued_csv: {out['rescued_csv']}")
        print(f"review_csv: {out['review_csv']}")
        print(f"cluster2_review_csv: {out['cluster2_review_csv']}")
        print(f"quality_csv: {out['quality_csv']}")
        print(f"report_csv: {out['report_csv']}")
        print(f"report_json: {out['report_json']}")
        print(f"candidate_rescued_unique_epics: {out['candidate_rescued_unique_epics']}")
        print(f"validated_rescued_unique_epics: {out['validated_rescued_unique_epics']}")
        return
    K2CampaignRunner().run()


def largeWindowMain():
    largeWindowModel = LargeWindowCnnModel()
    #largeWindowModel.build_model(mission="tess",neg_pos_ratio= 3, do_hard_neg=True)
   # largeWindowModel.build_model(mission="kepler",neg_pos_ratio= 7 , do_hard_neg=False)
    largeWindowModel.build_model(mission="k2", neg_pos_ratio=2,do_hard_neg=False)

if __name__ == "__main__":
    main()
