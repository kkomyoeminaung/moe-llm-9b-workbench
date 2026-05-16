import { execSync } from "child_process";
try {
  console.log(execSync("which python").toString());
} catch (e) { console.log("no python"); }
try {
  console.log(execSync("which python3").toString());
} catch (e) { console.log("no python3"); }
try {
  console.log(execSync("python3 -m pip --version").toString());
} catch (e) { console.log("no python3 -m pip"); }

