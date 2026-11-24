# standin-oracles.jq -- for TESTS that need a lint-clean copy of the fixture.
#
# Every permitted draft depth needs a pinned per-depth timed oracle per live
# golden (speculative_oracles.<prompt>.mtp<N>; tools/lint-benchmark-manifest.py
# check_live_goldens). Those goldens are recorded on a box with the pinned
# drafter, so until they are, the committed fixture does not carry them and the
# lint names each one: that lint run is the ONE place the gap is reported.
# A test that checks some OTHER property of the lint runs it on a copy of the
# fixture with this filter applied: a shape-valid stand-in pin for each oracle
# the fixture does not pin yet. A pin the fixture does carry is kept as is.
#
#   jq -f tools/testlib/standin-oracles.jq fixtures/muse_glimmer_30b_cuda_v1_track.json
(.timed_prompt_pool[0].r2_path | sub("/[^/]*$"; "")) as $prefix
| ([range(64)] | map("0") | join("")) as $zero
| (.mtp_head.permitted_draft_depths // []) as $depths
| reduce (.live_goldens // [])[] as $name (.;
    reduce $depths[] as $d (.;
      if (.speculative_oracles[$name]["mtp\($d)"] // null) != null then .
      else .speculative_oracles[$name]["mtp\($d)"] =
        {r2_path: "\($prefix)/\($name).mtp\($d).golden.json", sha256: $zero, bytes: 1}
      end))
