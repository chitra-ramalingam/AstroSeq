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
from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis import (
    K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderCachedFailedDownstreamReport import (
    K2DetectorQualityGatedBroaderCachedFailedDownstreamReport,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderPostRescueFailureAnalysis import (
    K2DetectorQualityGatedBroaderPostRescueFailureAnalysis,
)
from src.Classifiers.K2.Batch.K2CachedFailedBroaderDownstreamRunner import K2CachedFailedBroaderDownstreamRunner
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidation import K2DetectorQualityGatedScaleValidation
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis import (
    K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis import (
    K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment import (
    K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation import (
    K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation,
)
from src.Classifiers.K2.Batch.K2KnownConfirmedFalseNegativeAudit import K2KnownConfirmedFalseNegativeAudit
from src.Classifiers.K2.Batch.K2ConfirmedPlanetRecallAudit import K2ConfirmedPlanetRecallAudit
from src.Classifiers.K2.Batch.K2ConfirmedPlanetCoverageAudit import K2ConfirmedPlanetCoverageAudit
from src.Classifiers.K2.Batch.K2StageBPopulationManifest import K2StageBPopulationManifest
from src.Classifiers.K2.Batch.K2StageCActionQueue import K2StageCActionQueue
from src.Classifiers.K2.Batch.K2StageDExecutionPackaging import K2StageDExecutionPackaging
from src.Classifiers.K2.Batch.K2StageDDeeperEvalRunner import K2StageDDeeperEvalRunner
from src.Classifiers.K2.Batch.K2StageDTop10InspectionPackage import K2StageDTop10InspectionPackage
from src.Classifiers.K2.Batch.K2StageEHighPriorityBatchPlan import K2StageEHighPriorityBatchPlan
from src.Classifiers.K2.Batch.K2StageE1HighPriorityRerank import K2StageE1HighPriorityRerank
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation
from src.Classifiers.K2.Batch.K2StageFBatchExecution import K2StageFBatchExecution
from src.Classifiers.K2.Batch.K2StageFBatch001bPackaging import K2StageFBatch001bPackaging
from src.Classifiers.K2.Batch.K2StageFBatch001bExecution import K2StageFBatch001bExecution
from src.Classifiers.K2.Batch.K2StageFBatch001Audit import K2StageFBatch001Audit
from src.Classifiers.K2.Batch.K2StageGHighPriorityWhitenessAudit import K2StageGHighPriorityWhitenessAudit
from src.Classifiers.K2.Batch.K2StageHWhitenessPolicyDiagnosis import K2StageHWhitenessPolicyDiagnosis
from src.Classifiers.K2.Batch.K2StageLPostPatchCalibrationRerun import K2StageLPostPatchCalibrationRerun
from src.Classifiers.K2.Batch.K2StageRAutoLabeler import K2StageRAutoLabeler
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
    if len(argv) > 0 and argv[0] == "k2_stage_l_postpatch_calibration_rerun":
        out = K2StageLPostPatchCalibrationRerun.run_cli(argv=argv[1:])
        print(f"results_csv: {out['results_csv']}")
        print(f"summary_csv: {out['summary_csv']}")
        print(f"audit_csv: {out['audit_csv']}")
        print(f"rows_attempted: {out['rows_attempted']}")
        print(f"rows_completed: {out['rows_completed']}")
        print(f"rows_failed: {out['rows_failed']}")
        print(f"runtime_seconds: {out['runtime_seconds']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_r_label_events":
        out = K2StageRAutoLabeler.run_cli(argv=argv[1:])
        print(f"input_csv: {out['input_csv']}")
        print(f"output_csv: {out['output_csv']}")
        print(f"rows_input: {out['rows_input']}")
        print(f"rows_output: {out['rows_output']}")
        label_counts = out.get("label_counts", {})
        if isinstance(label_counts, dict):
            counts_text = " | ".join([f"{k}:{v}" for k, v in label_counts.items()])
            print(f"label_counts: {counts_text}")
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
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_broader_winner_downstream_analysis":
        out = K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis.run_cli(argv=argv[1:])
        print(f"detector_quality_gated_broader_winner_downstream_analysis.csv: {out['analysis_csv']}")
        print(f"detector_quality_gated_broader_winner_downstream_rollup.csv: {out['analysis_rollup_csv']}")
        print(f"detector_quality_gated_broader_real_rescues.csv: {out['real_rescues_csv']}")
        print(f"winners_total: {out['winners_total']}")
        print(f"real_rescues: {out['real_rescues']}")
        print(f"detector_only_gains: {out['detector_only_gains']}")
        print(f"still_blocked: {out['still_blocked']}")
        top_failure_reasons = out.get("top_failure_reasons", {})
        if isinstance(top_failure_reasons, dict) and len(top_failure_reasons) > 0:
            failure_text = " | ".join([f"{k}:{v}" for k, v in top_failure_reasons.items()])
            print(f"top_10_failure_reasons: {failure_text}")
        else:
            print("top_10_failure_reasons: none")
        rescue_bins = out.get("rescue_counts_by_period_bin", {})
        if isinstance(rescue_bins, dict) and len(rescue_bins) > 0:
            rescue_text = " | ".join([f"{k}:{v}" for k, v in rescue_bins.items()])
            print(f"rescue_counts_by_period_bin: {rescue_text}")
        else:
            print("rescue_counts_by_period_bin: none")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_broader_cached_failed_downstream_report":
        out = K2DetectorQualityGatedBroaderCachedFailedDownstreamReport.run_cli(argv=argv[1:])
        print(f"detector_quality_gated_broader_downstream_summary.csv: {out['summary_csv']}")
        print(f"detector_quality_gated_broader_quarantined_winners.csv: {out['quarantined_winners_csv']}")
        print(f"detector_quality_gated_broader_best_winners.csv: {out['best_winners_csv']}")
        print(f"winners_total_unique: {out['winners_total_unique']}")
        print(f"winners_in_best_only: {out['winners_in_best_only']}")
        print(f"winners_in_quarantine_only: {out['winners_in_quarantine_only']}")
        print(f"winners_in_both: {out['winners_in_both']}")
        print(f"winners_in_neither: {out['winners_in_neither']}")
        print(
            "corrected_downstream_conversion_rate: "
            f"{out['corrected_downstream_conversion_rate']:.6f} "
            f"({out['corrected_downstream_conversion_numerator']}/{out['corrected_downstream_conversion_denominator']})"
        )
        top_failure_reasons = out.get("top_failure_reasons", {})
        for reason_column in ["failure_category", "shortlist_rejection_reason", "terminal_reason"]:
            reason_map = top_failure_reasons.get(reason_column, {})
            if isinstance(reason_map, dict) and len(reason_map) > 0:
                reason_text = " | ".join([f"{k}:{v}" for k, v in reason_map.items()])
                print(f"top_{reason_column}: {reason_text}")
            else:
                print(f"top_{reason_column}: none")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_broader_post_rescue_failure_analysis":
        out = K2DetectorQualityGatedBroaderPostRescueFailureAnalysis.run_cli(argv=argv[1:])
        print(f"detector_quality_gated_broader_post_rescue_failure_analysis.csv: {out['analysis_csv']}")
        print(f"detector_quality_gated_broader_post_rescue_failure_rollup.csv: {out['rollup_csv']}")
        print(f"quarantined_winners_total: {out['quarantined_winners_total']}")
        bucket_counts = out.get("bucket_counts", {})
        for bucket in [
            "true insufficient signal",
            "likely recoverable with looser cluster/period policy",
            "likely recoverable with histogram handling changes",
            "likely unrecoverable / noise",
        ]:
            print(f"{bucket}: {bucket_counts.get(bucket, 0)}")
        print(f"recommended_next_lever: {out['recommended_next_lever']}")
        print(f"recommendation_rationale: {out['recommendation_rationale']}")
        return
    if len(argv) > 0 and argv[0] == "k2_cached_failed_broader_downstream":
        out = K2CachedFailedBroaderDownstreamRunner.run_cli(argv=argv[1:])
        print(f"shards_root: {out['shards_root']}")
        print(f"shard_count: {out['shard_count']}")
        print(f"out_dir: {out['out_dir']}")
        print(f"merged_batch_results.csv: {out['merged_batch_csv']}")
        print(f"downstream_input_shards.csv: {out['input_manifest_csv']}")
        print(f"shortlist_top_shape.csv: {out['shortlist_top_shape_csv']}")
        print(f"shortlist_top_shape_for_period.csv: {out['shortlist_top_shape_for_period_csv']}")
        print(f"period_shortlist_summary.csv: {out['period_shortlist_summary_csv']}")
        print(f"period_shortlist_summary_unique_epicP.csv: {out['period_shortlist_summary_unique_epicp_csv']}")
        print(f"period_shortlist_summary_validated_only.csv: {out['period_shortlist_summary_validated_only_csv']}")
        print(f"period_shortlist_best.csv: {out['period_shortlist_best_csv']}")
        print(f"period_shortlist_quarantine.csv: {out['period_shortlist_quarantine_csv']}")
        print(f"period_shortlist_diagnostics.csv: {out['period_shortlist_diagnostics_csv']}")
        print(f"epic_funnel_reasons.csv: {out['epic_funnel_reasons_csv']}")
        print(f"period_hist_summary_vs_best_counts.csv: {out['period_hist_counts_csv']}")
        print(f"validation_enabled: {out['validation_enabled']}")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_scale_validation":
        out = K2DetectorQualityGatedScaleValidation.run_cli(argv=argv[1:])
        print(f"sampled_epic_manifest.csv: {out['sample_manifest_csv']}")
        print(f"paired_detector_comparison.csv: {out['paired_detector_comparison_csv']}")
        print(f"downstream_summary.csv: {out['downstream_summary_csv']}")
        print(f"go_no_go_report.csv: {out['go_no_go_report_csv']}")
        print(f"go_no_go_report.txt: {out['go_no_go_report_txt']}")
        print(f"final_sample_n: {out['final_sample_n']}")
        print(f"observed_winners: {out['observed_winners']}")
        print(f"downstream_conversion_rate: {out['downstream_conversion_rate']:.6f}")
        print(
            f"downstream_conversion_ci: "
            f"[{out['downstream_conversion_ci_low']:.6f}, {out['downstream_conversion_ci_high']:.6f}]"
        )
        print(f"final_recommendation: {out['final_recommendation']}")
        print(f"last_expansion_reason: {out['last_expansion_reason']}")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_scale_validation_post_hold_failure_analysis":
        out = K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis.run_cli(argv=argv[1:])
        print(f"scale_validation_post_hold_quarantined_winners_analysis.csv: {out['analysis_csv']}")
        print(f"scale_validation_post_hold_quarantined_winners_rollup.csv: {out['rollup_csv']}")
        print(f"quarantined_winners_total: {out['quarantined_winners_total']}")
        bucket_counts = out.get("bucket_counts", {})
        for bucket in [
            "true insufficient signal",
            "histogram construction / handling",
            "cluster / period policy",
            "candidate filter policy",
            "something else",
        ]:
            print(f"{bucket}: {bucket_counts.get(bucket, 0)}")
        print(f"recommended_next_lever: {out['recommended_next_lever']}")
        print(f"recommendation_rationale: {out['recommendation_rationale']}")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_scale_validation_cluster_policy_analysis":
        out = K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis.run_cli(argv=argv[1:])
        print(f"scale_validation_cluster_policy_analysis.csv: {out['analysis_csv']}")
        print(f"scale_validation_cluster_policy_rollup.csv: {out['rollup_csv']}")
        print(f"cluster_policy_cases_total: {out['cluster_policy_cases_total']}")
        bucket_counts = out.get("bucket_counts", {})
        for bucket in [
            "supported MCC=2 carve-out candidate",
            "single-candidate near-miss",
            "three-event borderline",
            "two-event low-support",
        ]:
            print(f"{bucket}: {bucket_counts.get(bucket, 0)}")
        print(f"dominant_gate: {out['dominant_gate']}")
        print(f"recommended_smallest_safe_change: {out['recommended_smallest_safe_change']}")
        print(f"recommendation_rationale: {out['recommendation_rationale']}")
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_scale_validation_conditional_mcc2_experiment":
        out = K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment.run_cli(argv=argv[1:])
        print(f"conditional_mcc2_experiment_paired_downstream_analysis.csv: {out['experiment_analysis_csv']}")
        print(f"conditional_mcc2_experiment_comparison.csv: {out['comparison_csv']}")
        print(f"conditional_mcc2_experiment_summary.csv: {out['summary_csv']}")
        print(f"conditional_mcc2_experiment_go_no_go_report.csv: {out['go_no_go_csv']}")
        print(f"conditional_mcc2_experiment_go_no_go_report.txt: {out['go_no_go_txt']}")
        print(f"conditional_mcc2_experiment_decision_audit.csv: {out['decision_audit_csv']}")
        print(f"conditional_mcc2_experiment_decision_audit.txt: {out['decision_audit_txt']}")
        print(f"conditional_mcc2_experiment_next_limited_broader_validation_plan.txt: {out['broader_validation_plan_txt']}")
        print(f"baseline_winners_total: {out['baseline_metrics']['winners_total']}")
        print(f"baseline_winners_in_best: {out['baseline_metrics']['winners_in_best']}")
        print(f"baseline_winners_in_quarantine: {out['baseline_metrics']['winners_in_quarantine']}")
        print(f"baseline_downstream_conversion_rate: {out['baseline_metrics']['downstream_conversion_rate']:.6f}")
        print(f"baseline_quarantine_to_best_ratio: {out['baseline_metrics']['quarantine_to_best_ratio']:.6f}")
        print(f"baseline_final_recommendation: {out['baseline_metrics']['final_recommendation']}")
        print(f"experiment_winners_total: {out['experiment_metrics']['winners_total']}")
        print(f"experiment_winners_in_best: {out['experiment_metrics']['winners_in_best']}")
        print(f"experiment_winners_in_quarantine: {out['experiment_metrics']['winners_in_quarantine']}")
        print(f"experiment_downstream_conversion_rate: {out['experiment_metrics']['downstream_conversion_rate']:.6f}")
        print(f"experiment_quarantine_to_best_ratio: {out['experiment_metrics']['quarantine_to_best_ratio']:.6f}")
        print(f"experiment_final_recommendation: {out['experiment_metrics']['final_recommendation']}")
        print(f"paired_gain_cases: {out['paired_gain_cases']}")
        print(f"paired_regression_cases: {out['paired_regression_cases']}")
        print(f"harmful_regression_cases: {out['harmful_regression_cases']}")
        for row in out.get("decision_audit_rows", []):
            if str(row.get("section", "")) != "criterion":
                continue
            print(
                f"criterion[{row.get('criterion_group', '')}] {row.get('criterion_name', '')}: "
                f"observed={row.get('observed_value', '')} threshold={row.get('threshold', '')} "
                f"passed={row.get('passed', '')}"
            )
        for row in out.get("decision_audit_rows", []):
            if str(row.get("criterion_name", "")) == "hold_type":
                print(f"hold_type: {row.get('observed_value', '')}")
                print(f"final_recommendation_explanation: {row.get('explanation', '')}")
                break
        return
    if len(argv) > 0 and argv[0] == "k2_detector_quality_gated_conditional_mcc2_limited_broader_validation":
        out = K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation.run_cli(argv=argv[1:])
        print(f"limited_broader_validation_plan.txt: {out['plan_txt']}")
        print(f"operating_mode: {out['operating_mode']}")
        print(f"prepare_only: {out['prepare_only']}")
        print(f"run_command: {out['run_command']}")
        if not bool(out["prepare_only"]):
            runner_out = out["runner_out"]
            report_out = out["report_out"]
            print(f"period_shortlist_best.csv: {runner_out['period_shortlist_best_csv']}")
            print(f"period_shortlist_quarantine.csv: {runner_out['period_shortlist_quarantine_csv']}")
            print(f"epic_funnel_reasons.csv: {runner_out['epic_funnel_reasons_csv']}")
            print(f"detector_quality_gated_broader_downstream_summary.csv: {report_out['summary_csv']}")
            print(f"winners_total_unique: {report_out['winners_total_unique']}")
            print(f"winners_in_best_only: {report_out['winners_in_best_only']}")
            print(f"winners_in_quarantine_only: {report_out['winners_in_quarantine_only']}")
            print(f"winners_in_both: {report_out['winners_in_both']}")
            print(f"winners_in_neither: {report_out['winners_in_neither']}")
            print(
                "corrected_downstream_conversion_rate: "
                f"{report_out['corrected_downstream_conversion_rate']:.6f} "
                f"({report_out['corrected_downstream_conversion_numerator']}/"
                f"{report_out['corrected_downstream_conversion_denominator']})"
            )
        return
    if len(argv) > 0 and argv[0] == "k2_known_confirmed_false_negative_audit":
        out = K2KnownConfirmedFalseNegativeAudit.run_cli(argv=argv[1:])
        print(f"known_confirmed_false_negative_audit.csv: {out['analysis_csv']}")
        print(f"known_confirmed_false_negative_audit.txt: {out['report_txt']}")
        print(f"epic_count: {out['epic_count']}")
        for row in out.get("rows", []):
            print(f"epic: {row.get('epic_id', '')}")
            print(f"current_outcome_group: {row.get('current_outcome_group', '')}")
            print(f"primary_rejection_bucket: {row.get('primary_rejection_bucket', '')}")
            print(
                "policy_flags: "
                f"saved_default_survive={row.get('survives_under_saved_default_policy', False)} "
                f"conditional_mcc2_survive={row.get('survives_under_conditional_mcc2_carveout', False)} "
                f"larger_period_cap_survive={row.get('survives_under_larger_period_cap_from_existing_diagnostics', False)}"
            )
        return
    if len(argv) > 0 and argv[0] == "k2_confirmed_planet_recall_audit":
        out = K2ConfirmedPlanetRecallAudit.run_cli(argv=argv[1:])
        print(f"k2_confirmed_planet_recall_audit.csv: {out['audit_csv']}")
        print(f"k2_confirmed_planet_recall_rollup.csv: {out['rollup_csv']}")
        print(f"k2_confirmed_false_negatives.csv: {out['false_negatives_csv']}")
        print(f"nasa_confirmed_reference.csv: {out['reference_csv']}")
        print(f"confirmed_total: {out['confirmed_total']}")
        print(f"confirmed_in_best: {out['confirmed_in_best']}")
        print(f"confirmed_in_quarantine: {out['confirmed_in_quarantine']}")
        print(f"confirmed_detected_but_failed_downstream: {out['confirmed_detected_but_failed_downstream']}")
        print(f"confirmed_no_events_after_filters: {out['confirmed_no_events_after_filters']}")
        print(f"confirmed_not_seen: {out['confirmed_not_seen']}")
        print(f"confirmed_recall_best_only: {out['confirmed_recall_best_only']:.6f}")
        print(f"confirmed_recall_best_plus_quarantine: {out['confirmed_recall_best_plus_quarantine']:.6f}")
        top_failure_reasons = out.get("top_failure_reasons", {})
        if isinstance(top_failure_reasons, dict) and len(top_failure_reasons) > 0:
            failure_text = " | ".join([f"{k}:{v}" for k, v in top_failure_reasons.items()])
            print(f"top_failure_reasons_not_recovered_in_best: {failure_text}")
        else:
            print("top_failure_reasons_not_recovered_in_best: none")
        print(f"representative_false_negative_examples: {out.get('representative_examples', '')}")
        return
    if len(argv) > 0 and argv[0] == "k2_confirmed_planet_coverage_audit":
        out = K2ConfirmedPlanetCoverageAudit.run_cli(argv=argv[1:])
        print(f"k2_confirmed_planet_coverage_audit.csv: {out['audit_csv']}")
        print(f"k2_confirmed_planet_coverage_rollup.csv: {out['rollup_csv']}")
        print(f"confirmed_total: {out['confirmed_total']}")
        print(f"matched_to_processed_universe: {out['matched_to_processed_universe']}")
        print(f"not_processed: {out['not_processed']}")
        print(f"id_mismatch: {out['id_mismatch']}")
        print(f"outside_scope: {out['outside_scope']}")
        print(f"load_failed: {out['load_failed']}")
        print(f"final_dominant_coverage_blocker: {out['final_dominant_coverage_blocker']}")
        print(f"coverage_vs_science_conclusion: {out['coverage_vs_science_conclusion']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_b_population_manifest":
        out = K2StageBPopulationManifest.run_cli(argv=argv[1:])
        print(f"k2_stage_b_master_population_manifest.csv: {out['master_csv']}")
        print(f"k2_stage_b_master_unresolved_manifest.csv: {out['unresolved_csv']}")
        print(f"k2_stage_b_population_rollup.csv: {out['rollup_csv']}")
        print(f"total_relevant_epics: {out['total_relevant_epics']}")
        print(f"processed_universe_epics: {out['processed_universe_epics']}")
        print(f"known_confirmed_unique_epics: {out['known_confirmed_unique_epics']}")
        print(f"resolved_already_classified: {out['resolved_already_classified']}")
        print(f"known_confirmed_calibration_cases: {out['known_confirmed_calibration_cases']}")
        print(f"unresolved_needing_triage: {out['unresolved_needing_triage']}")
        print(f"load_failed_missing_light_curve: {out['load_failed_missing_light_curve']}")
        print(f"outside_current_scope: {out['outside_current_scope']}")
        print(f"exact_unresolved_manifest_path: {out['unresolved_csv']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_c":
        out = K2StageCActionQueue.run_cli(argv=argv[1:])
        print(f"k2_stage_c_action_queue.csv: {out['action_queue_csv']}")
        print(f"k2_stage_c_action_queue_summary.csv: {out['summary_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"process_now_count: {out['process_now_count']}")
        print(f"blocked_missing_light_curve_count: {out['blocked_missing_light_curve_count']}")
        print(f"outside_scope_count: {out['outside_scope_count']}")
        print(f"needs_manual_review_count: {out['needs_manual_review_count']}")
        print(f"rescue_path_candidate_count: {out['rescue_path_candidate_count']}")
        print(f"low_priority_or_defer_count: {out['low_priority_or_defer_count']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_d":
        out = K2StageDExecutionPackaging.run_cli(argv=argv[1:])
        print(f"k2_stage_d_process_now_high_priority.csv: {out['process_now_high_priority_csv']}")
        print(f"k2_stage_d_process_now_medium_priority.csv: {out['process_now_medium_priority_csv']}")
        print(f"k2_stage_d_rescue_candidates.csv: {out['rescue_candidates_csv']}")
        print(f"k2_stage_d_manual_review.csv: {out['manual_review_csv']}")
        print(f"k2_stage_d_deferred.csv: {out['deferred_csv']}")
        print(f"process_now_high_priority_count: {out['process_now_high_priority_count']}")
        print(f"process_now_medium_priority_count: {out['process_now_medium_priority_count']}")
        print(f"rescue_candidates_count: {out['rescue_candidates_count']}")
        print(f"manual_review_count: {out['manual_review_count']}")
        print(f"deferred_count: {out['deferred_count']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"ranking_logic: {out['ranking_logic']}")
        print(f"null_handling: {out['null_handling']}")
        missing = out["missing_ranking_fields"]
        print(
            "missing_ranking_fields: "
            f"best_depth_snr={missing['best_depth_snr']} "
            f"n_periods_proposed={missing['n_periods_proposed']} "
            f"n_events={missing['n_events']} "
            f"epic_id_norm={missing['epic_id_norm']} "
            f"rows_with_any_missing_ranking_field={missing['rows_with_any_missing_ranking_field']}"
        )
        return
    if len(argv) > 0 and argv[0] == "k2_stage_d_tier_a":
        out = K2StageDDeeperEvalRunner.run_cli(argv=argv[1:])
        print(f"k2_stage_d_tier_a_results.csv: {out['output_csv']}")
        print(f"rows_input: {out['rows_input']}")
        print(f"rows_output: {out['rows_output']}")
        print(f"pass_count: {out['pass_count']}")
        print(f"hold_count: {out['hold_count']}")
        print(f"fail_count: {out['fail_count']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_d_top10_inspection":
        out = K2StageDTop10InspectionPackage.run_cli(argv=argv[1:])
        print(f"stage_d_top10_dir: {out['out_dir']}")
        print(f"k2_stage_d_top10_inspection_index.csv: {out['index_csv']}")
        print(f"rows_input: {out['rows_input']}")
        print(f"top_n: {out['top_n']}")
        print(f"rows_output: {out['rows_output']}")
        print(f"top_epics: {' | '.join(out['top_epics'])}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_e":
        out = K2StageEHighPriorityBatchPlan.run_cli(argv=argv[1:])
        print(f"k2_stage_e_high_priority_batch_plan.csv: {out['batch_plan_csv']}")
        print(f"total_rows: {out['total_rows']}")
        print(f"batch_size: {out['batch_size']}")
        print(f"total_batches: {out['total_batches']}")
        print(f"rows_per_batch_summary: {out['rows_per_batch_summary']}")
        print(f"first_10_epics_batch_1: {' | '.join(out['first_10_epics_batch_1'])}")
        print(f"recommended_batch_size: {out['recommended_batch_size']}")
        print(f"batch_size_rationale: {out['batch_size_rationale']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_e1":
        out = K2StageE1HighPriorityRerank.run_cli(argv=argv[1:])
        print(f"k2_stage_e1_high_priority_rerank_preview.csv: {out['preview_csv']}")
        print(f"k2_stage_e1_high_priority_rerank_summary.csv: {out['summary_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"whiteness_proxy_coverage: {out['whiteness_proxy_coverage']}")
        print(f"old_top100_move_out: {out['old_top100_move_out']}")
        print(f"new_top100_enter: {out['new_top100_enter']}")
        print(
            "reranked_batch_001_materially_different: "
            f"{out['reranked_batch_001_materially_different']}"
        )
        print(f"recommendation: {out['recommendation']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_f_followup":
        out = K2StageFFollowupValidation.run_cli(argv=argv[1:])
        print(f"k2_stage_f_followup_validation.csv: {out['output_csv']}")
        print(f"stage_f_followup_dir: {out['out_dir']}")
        print(f"rows_input: {out['rows_input']}")
        print(f"rows_output: {out['rows_output']}")
        print(f"label_counts: {out['label_counts']}")
        print(f"planet_like_epics: {' | '.join(out['planet_like_epics'])}")
        print(f"hold_epics: {' | '.join(out['hold_epics'])}")
        print(f"likely_eb_epics: {' | '.join(out['likely_eb_epics'])}")
        print(f"reject_epics: {' | '.join(out['reject_epics'])}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_f":
        out = K2StageFBatchExecution.run_cli(argv=argv[1:])
        print(f"k2_stage_f_batch_001_results.csv: {out['results_csv']}")
        print(f"k2_stage_f_batch_001_summary.csv: {out['summary_csv']}")
        print(f"rows_attempted: {out['rows_attempted']}")
        print(f"rows_completed: {out['rows_completed']}")
        print(f"rows_failed: {out['rows_failed']}")
        print(f"rows_with_candidate_signal: {out['rows_with_candidate_signal']}")
        print(f"rows_without_candidate_signal: {out['rows_without_candidate_signal']}")
        print(f"rows_flagged_for_manual_review: {out['rows_flagged_for_manual_review']}")
        print(f"rows_requiring_rescue_followup: {out['rows_requiring_rescue_followup']}")
        print(f"command_used: {out['command_used']}")
        print(f"runtime_notes: {out['runtime_notes']}")
        print(f"failure_modes_encountered: {out['failure_modes_encountered']}")
        print(f"representative_for_batch_002: {out['representative_for_batch_002']}")
        print(f"representative_note: {out['representative_note']}")
        print(f"stage_f_run_dir: {out['stage_f_run_dir']}")
        print(f"pipeline_batch_results_csv: {out['pipeline_batch_results_csv']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_f_001b":
        out = K2StageFBatch001bPackaging.run_cli(argv=argv[1:])
        print(f"k2_stage_f_batch_001b_input.csv: {out['input_csv']}")
        print(f"k2_stage_f_batch_001b_plan_summary.csv: {out['plan_summary_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"first_10_epics: {' | '.join(out['first_10_epics'])}")
        print(f"median_whiteness_proxy_value: {out['median_whiteness_proxy_value']}")
        print(f"median_saved_triage_step_score: {out['median_saved_triage_step_score']}")
        print(f"median_n_events: {out['median_n_events']}")
        print(f"median_best_depth_snr: {out['median_best_depth_snr']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_f_001b_run":
        out = K2StageFBatch001bExecution.run_cli(argv=argv[1:])
        print(f"k2_stage_f_batch_001b_results.csv: {out['results_csv']}")
        print(f"k2_stage_f_batch_001b_summary.csv: {out['summary_csv']}")
        print(f"rows_attempted: {out['rows_attempted']}")
        print(f"rows_completed: {out['rows_completed']}")
        print(f"rows_failed: {out['rows_failed']}")
        print(f"rows_with_candidate_signal: {out['rows_with_candidate_signal']}")
        print(f"rows_without_candidate_signal: {out['rows_without_candidate_signal']}")
        print(f"rows_flagged_for_manual_review: {out['rows_flagged_for_manual_review']}")
        print(f"rows_requiring_rescue_followup: {out['rows_requiring_rescue_followup']}")
        print(f"final_label_counts: {out['final_label_counts']}")
        print(f"final_label_reason_counts: {out['final_label_reason_counts']}")
        print(f"command_used: {out['command_used']}")
        print(f"runtime_notes: {out['runtime_notes']}")
        print(f"failure_modes_encountered: {out['failure_modes_encountered']}")
        comparison = out['comparison']
        print(f"original_batch_001_noisy_trash_count: {comparison['original_noisy_trash_count']}")
        print(f"patched_batch_001b_noisy_trash_count: {comparison['current_noisy_trash_count']}")
        print(f"patched_batch_001b_non_noisy_count: {comparison['current_non_noisy_count']}")
        print(f"whiteness_rejection_frequency_improved: {comparison['whiteness_rejection_frequency_improved']}")
        print(f"enough_evidence_to_proceed_to_later_batches: {comparison['enough_evidence_to_proceed']}")
        print(f"comparison_note: {comparison['comparison_note']}")
        print(f"stage_f_run_dir: {out['stage_f_run_dir']}")
        print(f"pipeline_batch_results_csv: {out['pipeline_batch_results_csv']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_f_audit":
        out = K2StageFBatch001Audit.run_cli(argv=argv[1:])
        print(f"k2_stage_f_batch_001_audit.csv: {out['audit_csv']}")
        print(f"k2_stage_f_batch_001_audit_summary.csv: {out['audit_summary_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"label_counts: {out['label_counts']}")
        print(f"label_reason_counts: {out['label_reason_counts']}")
        print(
            "upstream_triage_usable_true_but_final_noisy_trash: "
            f"{out['upstream_triage_usable_true_but_final_noisy_trash']}"
        )
        print(
            "rejection_dominated_by_one_single_gate: "
            f"{out['rejection_dominated_by_one_single_gate']}"
        )
        print(f"dominant_gate_reason: {out['dominant_gate_reason']}")
        print(f"batch_002_likely_to_behave_similarly: {out['batch_002_likely_to_behave_similarly']}")
        print(f"recommendation: {out['recommendation']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_g":
        out = K2StageGHighPriorityWhitenessAudit.run_cli(argv=argv[1:])
        print(f"k2_stage_g_high_priority_whiteness_audit.csv: {out['audit_csv']}")
        print(f"k2_stage_g_high_priority_whiteness_audit_summary.csv: {out['summary_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"proxy_coverage: {out['proxy_coverage']}")
        print(f"whiteness_risk_bucket_counts: {out['whiteness_risk_bucket_counts']}")
        print(f"rows_with_n_periods_proposed_gt0: {out['rows_with_n_periods_proposed_gt0']}")
        print(f"likely_low_yield_fraction: {out['likely_low_yield_fraction']}")
        print(f"recommendation: {out['recommendation']}")
        return
    if len(argv) > 0 and argv[0] == "k2_stage_h":
        out = K2StageHWhitenessPolicyDiagnosis.run_cli(argv=argv[1:])
        print(f"k2_stage_h_whiteness_policy_diagnosis.csv: {out['diagnosis_csv']}")
        print(f"k2_stage_h_whiteness_policy_diagnosis_summary.csv: {out['summary_csv']}")
        print(f"rows_total: {out['rows_total']}")
        print(f"saved_proxy_pass_runtime_fail_count: {out['saved_proxy_pass_runtime_fail_count']}")
        print(f"same_definition_count: {out['same_definition_count']}")
        print(f"step_score_exact_match_count: {out['step_score_exact_match_count']}")
        print(f"runtime_whiteness_zero_count: {out['runtime_whiteness_zero_count']}")
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
