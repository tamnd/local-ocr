#!/usr/bin/env bash
#
# Restart a reader that has stopped generating.
#
# The unit next door has Restart=on-failure, which covers a reader that dies and
# does not cover a reader that stops working without dying. Both have happened.
#
#   07:08:40, one morning: the server shut down cleanly and nothing noticed for
#   42 minutes. Both round robin loops kept handing windows to a dead port and
#   every page came back "the model will not return this page, handed back
#   unread", which reads exactly like a page the reader refuses. The pages were
#   fine. The server was gone. Restart=on-failure would not have covered this
#   one either, because the exit was clean.
#
#   17:46, 22 August: the engine wedged holding all 20988 MiB of the card at
#   nought per cent utilisation. evt-i-v-fr sat in "reading" for an hour and
#   four minutes. The process was alive the whole time, so systemd had nothing
#   to restart and a liveness check on the process would have reported a healthy
#   reader. GET /v1/models went on answering 200 for another 56 minutes after
#   the engine stopped, because listing models is served off a dict and never
#   goes near the engine loop.
#
# Hence a check that generates a token, and a restart that does not assume the
# old process will get out of the way by itself.
#
# Run it on the reader host, or from anywhere with READER_HOST set to an ssh
# destination. The fleet runs the second form from the laptop that drives the
# grind, because that laptop is the machine that notices when pages stop
# arriving and it is the one place that is always up.
#
#   ./deploy/reader-watch.sh                      # on the reader host
#   READER_HOST=gpc ./deploy/reader-watch.sh      # over ssh
#
set -u

PORT=${READER_PORT:-8801}
MODEL=${READER_MODEL:-reader-a}
UNIT=${READER_UNIT:-local-ocr-reader@${MODEL}}
START=${READER_START:-$HOME/start-reader.sh}
HOST=${READER_HOST:-}
EVERY=${READER_EVERY:-60}

# Three and not two. A probe queues behind whatever the reader is already
# serving, the fleet serves 16 sequences at a time, and a slow answer under a
# full batch is not a dead server. Three misses puts the restart between two
# minutes and five and a quarter after generation stops, depending on whether
# the probes time out or fail at once, against the hour and four minutes it took
# to notice by hand.
STRIKES=${READER_STRIKES:-3}
TIMEOUT=${READER_TIMEOUT:-45}

# Four minutes to load and compile, and some slack. Polling through a start just
# burns strikes on a server that is doing what it was told.
SETTLE=${READER_SETTLE:-300}

say() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Everything below runs through this, so that the local and the ssh forms are
# the same script rather than two scripts that drift apart.
run() {
  if [ -n "$HOST" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$1" 2>/dev/null
  else
    bash -c "$1" 2>/dev/null
  fi
}

# One token off the real engine, which is the only part of the server that has
# been observed to fail on its own. Prints 1 when a completion came back.
probe() {
  run "curl -s -m $TIMEOUT http://127.0.0.1:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"$MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}' \
      | grep -c finish_reason"
}

# SIGKILL and not SIGTERM. A wedged engine was sent a SIGTERM on 22 August and
# was still holding 20988 MiB fifteen seconds later, and the unit's KillSignal
# assumes a server well enough to act on a signal. Then wait for the memory to
# come back, because the replacement loads into whatever the old one still
# holds and a second server on a 24 GB card gets an allocator error four minutes
# into loading its weights.
#
# The bracket in "[v]llm" is load bearing. pkill -f 'vllm serve' run over ssh
# matches the shell wrapper that carries the pattern in its own command line,
# and kills the session instead of the server.
clear_card() {
  run '
    pids=$(pgrep -f "[v]llm" | tr "\n" " ")
    if [ -n "$pids" ]; then kill -9 $pids 2>/dev/null; fi
    used=unknown
    for _ in $(seq 1 30); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      case $used in ""|*[!0-9]*) break ;; esac
      [ "$used" -lt 1000 ] && break
      sleep 2
    done
    echo "$used"'
}

# Under systemd where the unit is installed, by hand where it is not. The fleet
# host runs the reader from a start script because the unit wants a root install
# and the card belongs to a desktop that also does other things; both paths are
# here so that installing the unit later does not mean editing this file.
start() {
  if [ "$(run "systemctl is-enabled $UNIT 2>/dev/null || true")" = "enabled" ]; then
    run "systemctl restart $UNIT"
    echo "systemd restarted $UNIT"
  else
    # Through the start script and not with the vllm command spelled out here,
    # because the command spelled out here lost HF_HUB_OFFLINE once. Without it
    # the engine looks up weights it already has, the lookup goes out over IPv6
    # and hangs, and the restart leaves 48 MiB on the card and no server at all.
    run "setsid nohup $START > /tmp/start-reader.out 2>&1 < /dev/null &"
    echo "started $START"
  fi
}

say "watching $MODEL on ${HOST:-localhost}:$PORT, $STRIKES misses of ${TIMEOUT}s to restart"

misses=0
while true; do
  if [ "$(probe)" = "1" ]; then
    if [ "$misses" -gt 0 ]; then
      say "generating again after $misses missed probes"
    fi
    misses=0
  else
    misses=$((misses + 1))
    say "no completion on $PORT, miss $misses of $STRIKES"
    if [ "$misses" -ge "$STRIKES" ]; then
      say "card left at $(clear_card) MiB"
      say "$(start)"
      misses=0
      sleep "$SETTLE"
    fi
  fi
  sleep "$EVERY"
done
