#!/bin/bash
# on the notebook: kill any fhwrap python whose fh dump shows it stuck in model loading (.to / _materialize_copy) for 2+ dumps
while true; do
  for pid in $(pgrep -f "python fhwrap.py"); do
    f=/workspace/logs/fh_$pid.txt; [ -f $f ] || continue
    n=$(grep -c "Timeout (0:02:00)" $f)
    if [ "$n" -ge 1 ] && tail -40 $f | grep -q -E "_materialize_copy|in convert|load_model_and_tokenizer"; then
      if ! tail -40 $f | grep -q -E "generate|predict_local\" , line 1[0-9][0-9]"; then
        echo "$(date -u +%T) kill stalled load pid $pid" >> /workspace/logs/day2.log; kill $pid
      fi
    fi
  done
  sleep 20
  pgrep -f run_day2b.sh >/dev/null || break
done
