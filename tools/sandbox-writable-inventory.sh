#!/usr/bin/env bash
# sandbox-writable-inventory.sh -- every place on this box the sandbox user can
# keep data in, found AS that user, against the wipe list
# (docs/sandbox-design.md 6a, "Writable places"). TRUSTED: not in
# editablePaths.
#
# tools/sandbox-wipe.sh clears only the directories in the wipe list, and only
# what the sandbox user owns there. So any other place that user can write is a
# way to carry data from one job or leg to the next. Setup records the list
# once; this re-takes the inventory, as the sandbox user, every time it runs:
# the ranked preflight runs `check` before every job.
#
# One find runs as the sandbox user (the runner's existing sudo drop) over every
# local filesystem (findmnt types ext2/3/4, xfs, btrfs, f2fs, vfat, tmpfs, plus
# /dev/mqueue), never crossing a mount, skipping /proc, /sys and docker's and
# containerd's layer trees. -writable asks access(2) as that user, so a group
# grant or an ACL entry counts, not only a world-writable mode. It records:
#   D  a directory the sandbox user can write;
#   F  a regular file it can write that it does not own (the wipe never
#      removes such a file, so its content stays);
#   O  anything it owns (it can chmod and rewrite it);
#   B  a directory it can enter but not list (find cannot look inside).
#
#   check  refuses (exit 1) when
#          - a D is neither in the wipe list nor under a listed directory;
#          - there is any F;
#          - an O is outside every listed directory (inside one, the wipe
#            removes it).
#          A B the runner can list is looked into: its entries are handed to
#          the same find, as the sandbox user, by name. A B neither account can
#          list holds only what root put there (only root can add an entry to
#          it, unless the B itself is writable, which is a D); it is counted on
#          stderr, never refused.
#   list   prints every D outside SANDBOX_ROOT, one per line, for
#          tools/box-runner/setup-sandbox-user.sh to add to the wipe list.
#
# Only paths the sandbox user does not own are printed (at most five per
# kind): a name it chose could carry prompt text, so its own are counted.
#
# Env: SANDBOX_USER (default nemosub), SANDBOX_ETC (its wipe-dirs; or
# SANDBOX_WIPE_LIST), SANDBOX_ROOT (default /var/lib/muse-glimmer/sandbox).
# SANDBOX_INVENTORY_ROOTS (newline-separated) replaces the filesystem list, for
# tests only; the ranked preflight refuses it set.
# Exit: 0 clean; 1 refused; 2 the inventory could not be taken; 78 usage.
set -uo pipefail

mode="${1:-}"
[[ "${mode}" == check || "${mode}" == list ]] || { echo "usage: sandbox-writable-inventory.sh check|list" >&2; exit 78; }
U="${SANDBOX_USER:-nemosub}"
WIPE_LIST="${SANDBOX_WIPE_LIST:-${SANDBOX_ETC:-/etc/muse-glimmer-sandbox}/wipe-dirs}"
SROOT="${SANDBOX_ROOT:-/var/lib/muse-glimmer/sandbox}"
say() { echo "sandbox-writable-inventory.sh: $*" >&2; }

sudo -n -u "${U}" -- true >/dev/null 2>&1 || { say "'sudo -n -u ${U}' fails; no inventory"; exit 2; }
if [[ -n "${SANDBOX_INVENTORY_ROOTS:-}" ]]; then
  mapfile -t ROOTS <<<"${SANDBOX_INVENTORY_ROOTS}"
else
  command -v findmnt >/dev/null 2>&1 || { say "findmnt is required"; exit 2; }
  mapfile -t ROOTS < <(findmnt -rn -o TARGET -t ext2,ext3,ext4,xfs,btrfs,f2fs,vfat,tmpfs; \
                       [[ -d /dev/mqueue ]] && echo /dev/mqueue)
fi
[[ "${#ROOTS[@]}" -gt 0 && -n "${ROOTS[0]}" ]] || { say "no filesystem to search"; exit 2; }

declare -A LISTED=()
if [[ "${mode}" == check ]]; then
  [[ -f "${WIPE_LIST}" && -r "${WIPE_LIST}" ]] || { say "no wipe list ${WIPE_LIST}"; exit 2; }
  while IFS= read -r d; do LISTED["${d%/}"]=1; done < <(grep '^/' "${WIPE_LIST}")
fi
# covered PATH -- PATH is a listed directory or under one.
covered() {
  local p="$1"
  while [[ -n "${p}" && "${p}" != / ]]; do
    [[ -n "${LISTED[${p}]:-}" ]] && return 0
    p="${p%/*}"
  done
  [[ -n "${LISTED[/]:-}" ]]
}

# scan START... -- the find, as the sandbox user; NUL-terminated "K<TAB>path"
# records, then one "END<TAB>rc" record. rc 1 is find's answer for a directory
# it may not read, which is expected (nothing below it is reachable by name
# from a listing); any other non-zero rc is a failed inventory.
scan() {
  # shellcheck disable=SC2016  # expands in the inner shell
  sudo -n -u "${U}" -- sh -c 'u="$1"; shift
find -P "$@" -xdev \( -path /proc -o -path /sys -o -path /var/lib/docker -o -path /var/lib/containerd \) -prune -o \
  \( -user "$u" -printf "O\t%p\0" \) , \( -type d -writable -printf "D\t%p\0" \) , \
  \( -type f -writable ! -user "$u" -printf "F\t%p\0" \) , \( -type d -executable ! -readable -printf "B\t%p\0" \) 2>/dev/null
printf "END\t%s\0" "$?"' sh "${U}" "$@"
}

declare -A SEEN_B=()
bad_d=(); bad_f=(); own_out=0; all_d=(); blind_unlisted=0
queue=("${ROOTS[@]}")
while [[ "${#queue[@]}" -gt 0 ]]; do
  next=(); ended=""
  while IFS= read -r -d '' rec; do
    k="${rec%%$'\t'*}"; p="${rec#*$'\t'}"
    case "${k}" in
      END) ended="${p}" ;;
      D) all_d+=("${p}"); covered "${p}" || bad_d+=("${p}") ;;
      F) bad_f+=("${p}") ;;
      O) covered "${p}" || own_out=$((own_out + 1)) ;;
      B)
        [[ -n "${SEEN_B[${p}]:-}" ]] && continue
        SEEN_B["${p}"]=1
        if entries="$(find -P "${p}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | tr '\0' '\n')" && [[ -r "${p}" ]]; then
          while IFS= read -r e; do [[ -n "${e}" ]] && next+=("${e}"); done <<<"${entries}"
        else
          blind_unlisted=$((blind_unlisted + 1))
        fi ;;
    esac
  done < <(scan "${queue[@]}")
  [[ "${ended}" == 0 || "${ended}" == 1 ]] || { say "the find as ${U} did not finish (rc '${ended}'); no inventory"; exit 2; }
  queue=("${next[@]}")
done

if [[ "${mode}" == list ]]; then
  for d in "${all_d[@]}"; do [[ "${d}" == "${SROOT}" || "${d}" == "${SROOT}"/* ]] || printf '%s\n' "${d}"; done \
    | awk '!seen[$0]++'
  exit 0
fi

(( blind_unlisted == 0 )) \
  || say "note: ${blind_unlisted} director(ies) ${U} can enter but neither ${U} nor $(id -un) can list were not looked into (only root can add to them)"
rc=0
# A name the sandbox user chose is never printed; only others' paths are.
show() { local p; for p in "${@:1:5}"; do [[ "$(stat -c %U -- "${p}" 2>/dev/null)" == "${U}" ]] || say "  ${p}"; done; }
if (( ${#bad_d[@]} > 0 )); then
  say "REFUSING -- ${U} can write ${#bad_d[@]} director(ies) outside the wipe list ${WIPE_LIST}; re-run tools/box-runner/setup-sandbox-user.sh as root, or remove the grant:"
  show "${bad_d[@]}"; rc=1
fi
if (( ${#bad_f[@]} > 0 )); then
  say "REFUSING -- ${U} can write ${#bad_f[@]} file(s) it does not own (the wipe never removes them); remove its write access:"
  show "${bad_f[@]}"; rc=1
fi
if (( own_out > 0 )); then
  say "REFUSING -- ${U} owns ${own_out} path(s) outside every wipe-list directory (names not printed); remove them as root"
  rc=1
fi
(( rc == 0 )) && echo "sandbox-writable-inventory.sh: every place ${U} can write is in the wipe list (${#all_d[@]} writable director(ies) found as ${U})"
exit "${rc}"
