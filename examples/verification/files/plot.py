import matplotlib
matplotlib.use('Svg')
import matplotlib.pyplot as plt

TIMEOUT = 300
INSTANCES = 464

def get_runtimes(filename):
  runtimes = []
  with open(filename, 'r') as f:
    for l in f.readlines():
      verification_result = l.split(',')[4]
      if verification_result in ["timeout", "no_result_in_file", 'run_instance_timeout']:
        runtime = TIMEOUT
      else:
        assert "unsat" in l, l
        runtime = min(TIMEOUT, float(l.split(',')[-1]))
      runtimes.append(runtime)
  assert len(runtimes) == INSTANCES
  return runtimes

def sort_and_filter(runtimes):
  runtimes.sort()
  # The first 348 instances are solved instantly, so INVPROP cannot improve the runtime further
  # The next 57 instances are sped up by INVPROP
  # The remaining instances time out even with INVPROP
  runtimes = runtimes[348:]
  runtimes = runtimes[:58]
  return runtimes

orig_runtimes = get_runtimes('results/orig.csv')
invprop_runtimes = get_runtimes('results/invprop.csv')

orig_runtimes = sort_and_filter(orig_runtimes)
invprop_runtimes = sort_and_filter(invprop_runtimes)

xs = [x for x in range(len(orig_runtimes))]

ax = plt.gca()
ax.plot(xs, orig_runtimes, label="alpha-beta-CROWN")
ax.plot(xs, invprop_runtimes, label="alpha-beta-CROWN + INVPROP")
ax.set_xlabel("Number of instances verified")
ax.set_ylabel("Max runtime [s]")
ax.legend(loc="upper left")
ax.set_yscale('log')
ax.set_yticks([10, 100, TIMEOUT])
ax.set_yticklabels([10, 100, TIMEOUT])
plt.savefig('results/plot.png')