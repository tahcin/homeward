import faulthandler, os, runpy, sys
# Oct 17: host-to-GPU copies stalled at random (in from_pretrained and in model.to); copy without the SDMA engine.
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
f = open(f"/workspace/logs/fh_{os.getpid()}.txt", "w")
faulthandler.dump_traceback_later(120, repeat=True, file=f)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
