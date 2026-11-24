# serial-declaration.jq -- for TESTS that need a SERIAL declaration.
#
# The shipped mtp-head.manifest.json is the track's base, and the base may
# switch the pinned drafter on (the drafter track ships it at depth 4). A test
# whose case is "the serial declaration" must not copy the shipped file, or it
# tests whatever the base declares. It writes this one instead: the contract's
# own pin (source, repository, full revision, sha256 per file), drafter off.
#
#   jq -f tools/testlib/serial-declaration.jq fixtures/muse_glimmer_30b_cuda_v1_track.json
{version: 1, source: .mtp_head.source,
 upstream_model_id: .mtp_head.upstream_model_id, upstream_revision: .mtp_head.upstream_revision,
 sha256: ([.mtp_head.files[] | {key: .path, value: .sha256}] | from_entries),
 spec: {decoder: .mtp_head.decoder, enabled: false, num_speculative_tokens: 0}}
