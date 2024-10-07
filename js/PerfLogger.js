export default class PerfLogger {
  constructor() {
    this.timers = [];
  }

  log() {
    this.timers.push(performance.now());
  }

  print(doReset = false) {
    let s = "Performances:\n";

    if (this.timers.length > 2) {
      for (let i=1; i < this.timers.length; ++i) {
        const perf = (this.timers[i] - this.timers[i-1]).toFixed(2);
        s += `t${i}: ${perf}\n`;
      }
    }

    s += `total : ${(this.timers[this.timers.length-1] - this.timers[0]).toFixed(2)} ms\n`;
    console.log(s);

    if (doReset) {
      this.timers = [];
    }
  }
};
