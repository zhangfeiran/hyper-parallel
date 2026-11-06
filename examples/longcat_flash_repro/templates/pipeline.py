# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import json
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
phases = [
    ("component", ["bash", str(R / "run_formal_component.sh")]),
    ("trainer", ["bash", str(R / "run_formal_trainer.sh")]),
]
done = []
try:
    for name, command in phases:
        (R / "status.json").write_text(
            json.dumps({"status": "RUNNING", "phase": name, "completed": done}, indent=2) + "\n", encoding="utf-8"
        )
        print(name, "START", flush=True)
        t = time.time()
        subprocess.run(command, check=True)
        done.append({"phase": name, "seconds": time.time() - t})
        print(name, "COMPLETE", flush=True)
    (R / "status.json").write_text(
        json.dumps({"status": "COMPLETE", "completed": done}, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run([sys.executable, str(R / "aggregate.py")], check=True)
except BaseException as error:
    (R / "status.json").write_text(
        json.dumps({"status": "FAILED", "phase": name, "error": repr(error), "completed": done}, indent=2) + "\n",
        encoding="utf-8",
    )
    raise
