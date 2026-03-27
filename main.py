import sys

from src.Classifiers.K2.Pipeline.K2CampaignRunner import K2CampaignRunner
from src.Classifiers.K2.Pipeline.K2FailedDownloader import K2FailedDownloader
from src.Classifiers.K2.Pipeline.K2PosthocRanking import K2PosthocRanking
from src.Classifiers.K2.Pipeline.K2PosthocRetriage import K2PosthocRetriage
from src.Classifiers.K2.Pipeline.K2WhitenessRunner import K2WhitenessRunner
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2ShortlistPeriodCompare import K2ShortlistPeriodCompare
from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis
from src.Classifiers.K2.Batch.K2ShortlistDetectorModeAnalysis import K2ShortlistDetectorModeAnalysis
from src.Classifiers.K2.Batch.K2DetectorQualityGatedComparison import K2DetectorQualityGatedComparison
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
    if len(argv) > 0 and argv[0] == "k2_shortlist_recovery_mode_analysis":
        out = K2ShortlistRecoveryModeAnalysis.run_cli(argv=argv[1:])
        print(f"post_mcc_remaining_failures_by_reason.csv: {out['post_mcc_remaining_failures_by_reason_csv']}")
        print(f"post_mcc_remaining_failures_by_period_bin.csv: {out['post_mcc_remaining_failures_by_period_bin_csv']}")
        print(f"recovery_mode_comparison.csv: {out['recovery_mode_comparison_csv']}")
        print(f"rescued_by_mode.csv: {out['rescued_by_mode_csv']}")
        print(f"post_mcc_no_p_available_whiteness_diagnostics.csv: {out['post_mcc_no_p_available_whiteness_diagnostics_csv']}")
        print(f"no_p_available_upstream_blocker_summary.csv: {out['no_p_available_upstream_blocker_summary_csv']}")
        print(f"no_p_available_upstream_blocker_by_period_bin.csv: {out['no_p_available_upstream_blocker_by_period_bin_csv']}")
        print(f"no_upstream_events_detected_diagnostics.csv: {out['no_upstream_events_detected_diagnostics_csv']}")
        print(f"too_few_events_remaining_after_filtering_diagnostics.csv: {out['too_few_events_remaining_after_filtering_diagnostics_csv']}")
        print(f"first_failed_upstream_stage_summary.csv: {out['first_failed_upstream_stage_summary_csv']}")
        print(f"first_failed_upstream_stage_by_period_bin.csv: {out['first_failed_upstream_stage_by_period_bin_csv']}")
        print(f"event_detection_zero_events_diagnostics.csv: {out['event_detection_zero_events_diagnostics_csv']}")
        print(f"event_detection_insufficient_support_diagnostics.csv: {out['event_detection_insufficient_support_diagnostics_csv']}")
        print(f"suspected_zero_event_cause_summary.csv: {out['suspected_zero_event_cause_summary_csv']}")
        print(f"suspected_zero_event_cause_by_period_bin.csv: {out['suspected_zero_event_cause_by_period_bin_csv']}")
        print(f"suspected_insufficient_support_cause_summary.csv: {out['suspected_insufficient_support_cause_summary_csv']}")
        print(f"suspected_insufficient_support_cause_by_period_bin.csv: {out['suspected_insufficient_support_cause_by_period_bin_csv']}")
        print(f"remaining_top_failure_reasons: {out['remaining_top_failure_reasons']}")
        print(f"threshold_added_vs_mcc2: {out['threshold_added_vs_mcc2']}")
        print(f"period_bin_15_20_delta_vs_mcc2: {out['period_bin_15_20_delta_vs_mcc2']}")
        print(f"manual_review_delta_vs_mcc2: {out['manual_review_delta_vs_mcc2']}")
        print(f"no_p_whiteness_value_column: {out['no_p_whiteness_value_column']}")
        print(f"dominant_no_p_upstream_blocker: {out['dominant_no_p_upstream_blocker']}")
        print(f"dominant_first_failed_upstream_stage: {out['dominant_first_failed_upstream_stage']}")
        print(f"top_suspected_zero_event_cause: {out['top_suspected_zero_event_cause']}")
        print(f"top_suspected_insufficient_support_cause: {out['top_suspected_insufficient_support_cause']}")
        print(f"no_upstream_events_detected_count: {out['no_upstream_events_detected_count']}")
        print(f"too_few_events_remaining_after_filtering_count: {out['too_few_events_remaining_after_filtering_count']}")
        print(f"no_p_whiteness_related_blocker_count: {out['no_p_whiteness_related_blocker_count']}")
        print(f"events_filtered_to_zero_whiteness_related_count: {out['events_filtered_to_zero_whiteness_related_count']}")
        return
    if len(argv) > 0 and argv[0] == "k2_shortlist_detector_mode_analysis":
        out = K2ShortlistDetectorModeAnalysis.run_cli(argv=argv[1:])
        print(f"detector_mode_comparison.csv: {out['detector_mode_comparison_csv']}")
        print(f"rescued_by_detector_mode.csv: {out['rescued_by_detector_mode_csv']}")
        print(f"rescued_by_detector_mode_by_period_bin.csv: {out['rescued_by_detector_mode_by_period_bin_csv']}")
        print(f"detector_added_vs_mcc2: {out['detector_added_vs_mcc2']}")
        print(f"period_bin_15_20_delta_vs_mcc2: {out['period_bin_15_20_delta_vs_mcc2']}")
        print(f"zero_event_delta_vs_mcc2: {out['zero_event_delta_vs_mcc2']}")
        print(f"insufficient_support_delta_vs_mcc2: {out['insufficient_support_delta_vs_mcc2']}")
        print(f"manual_review_delta_vs_mcc2: {out['manual_review_delta_vs_mcc2']}")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_comparison":
        out = K2DetectorQualityGatedComparison.run_cli(argv=argv[1:])
        print(f"detector_quality_gated_comparison.csv: {out['out_csv']}")
        print(f"detector_quality_gated_epic_summary.csv: {out['epic_summary_csv']}")
        print(f"detector_quality_gated_rollup.csv: {out['rollup_csv']}")
        print(f"row_count: {out['row_count']}")
        print(f"epic_count: {out['epic_count']}")
        print(f"quality_gated_keeps_some_event_count_gain: {out['keeps_some_event_count_gain']}")
        print(f"quality_gated_event_gain_epic_count_vs_default: {out['qg_event_gain_epic_count_vs_default']}")
        print(f"quality_gated_event_gain_total_vs_default: {out['qg_event_gain_total_vs_default']}")
        print(f"quality_gated_improves_best_shape_score_on_any_epic: {out['improves_best_shape_score_any']}")
        print(
            "quality_gated_shape_improved_epic_count_vs_plain_experimental: "
            f"{out['qg_shape_improved_epic_count_vs_experimental']}"
        )
        print(f"quality_gated_improves_best_depth_snr_on_any_epic: {out['improves_best_depth_snr_any']}")
        print(
            "quality_gated_depth_improved_epic_count_vs_plain_experimental: "
            f"{out['qg_depth_improved_epic_count_vs_experimental']}"
        )
        print(f"quality_gated_event_delta_total_vs_plain_experimental: {out['qg_event_delta_total_vs_experimental']}")
        print(
            "quality_gated_event_loss_epic_count_vs_plain_experimental: "
            f"{out['qg_event_loss_epic_count_vs_experimental']}"
        )
        print(
            "experimental_extra_events_vs_default_count: "
            f"{out['experimental_extra_events_vs_default_count']}"
        )
        print(
            "quality_gated_extra_events_vs_default_count: "
            f"{out['quality_gated_extra_events_vs_default_count']}"
        )
        print(
            "any_best_shape_score_improvement_vs_default_count: "
            f"{out['any_best_shape_score_improvement_vs_default_count']}"
        )
        print(
            "any_best_depth_snr_improvement_vs_default_count: "
            f"{out['any_best_depth_snr_improvement_vs_default_count']}"
        )
        print(
            "plain_high_recall_regressed_quality_count: "
            f"{out['plain_high_recall_regressed_quality_count']}"
        )
        print(
            "quality_gated_avoided_plain_regression_count: "
            f"{out['quality_gated_avoided_plain_regression_count']}"
        )
        print(
            "quality_gated_looks_better_than_plain_detector_high_recall_experimental_for_scaling: "
            f"{out['looks_better_for_scaling']}"
        )
        print(f"recommendation: {out['recommendation']}")
        return
    K2CampaignRunner().run()


def largeWindowMain():
    largeWindowModel = LargeWindowCnnModel()
    #largeWindowModel.build_model(mission="tess",neg_pos_ratio= 3, do_hard_neg=True)
   # largeWindowModel.build_model(mission="kepler",neg_pos_ratio= 7 , do_hard_neg=False)
    largeWindowModel.build_model(mission="k2", neg_pos_ratio=2,do_hard_neg=False)

if __name__ == "__main__":
    main()
