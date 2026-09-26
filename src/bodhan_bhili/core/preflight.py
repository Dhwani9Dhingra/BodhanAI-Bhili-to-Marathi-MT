"""
Hardware, software, model-access and metric preflight.

This script deliberately runs BEFORE loading the 7.94B Bodhan weights.

The objective is to fail in seconds/minutes instead of discovering a
broken environment after a multi-gigabyte download.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from dataclasses import (
    asdict,
    dataclass,
)
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
)

from packaging.version import (
    Version,
)

from bodhan_bhili.core.config import (
    ExperimentConfig,
)
from bodhan_bhili.core.environment import (
    collect_gpu_info,
    package_version,
    recommended_compute_dtype,
)
from bodhan_bhili.core.logging import (
    get_logger,
)
from bodhan_bhili.core.paths import (
    ArtifactPaths,
)
from bodhan_bhili.core.serialization import (
    atomic_write_json,
)


class CheckStatus(
    str,
    Enum,
):
    """Possible preflight result states."""

    PASS = "PASS"

    WARN = "WARN"

    FAIL = "FAIL"


@dataclass
class CheckResult:
    """One independently testable environment condition."""

    name: str

    status: CheckStatus

    message: str

    details: dict[str, Any] | None = None


class PreflightRunner:
    """Execute every pre-training environment check."""

    def __init__(
        self,
        config: ExperimentConfig,
        paths: ArtifactPaths,
        require_data: bool = False,
    ):
        self.config = config

        self.paths = paths

        self.require_data = require_data

        self.logger = get_logger()

        self.results: list[
            CheckResult
        ] = []


    def _record(
        self,
        result: CheckResult,
    ) -> None:
        """Store and immediately log one result."""

        self.results.append(
            result
        )

        log_method = {
            CheckStatus.PASS:
                self.logger.info,

            CheckStatus.WARN:
                self.logger.warning,

            CheckStatus.FAIL:
                self.logger.error,
        }[
            result.status
        ]

        log_method(
            "[%s] %s — %s",
            result.status.value,
            result.name,
            result.message,
        )


    def _execute(
        self,
        name: str,
        check: Callable[
            [],
            CheckResult,
        ],
    ) -> None:
        """
        Execute a check without preventing later checks from running.

        This is important because one preflight run should reveal ALL
        configuration problems, not only the first problem.
        """

        try:
            result = check()

        except Exception as exc:
            result = CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                message=(
                    f"Unexpected error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        self._record(
            result
        )


    # ========================================================
    # PYTHON
    # ========================================================

    def _check_python(
        self,
    ) -> CheckResult:

        import platform

        installed = Version(
            platform.python_version()
        )

        minimum = Version(
            self.config
            .preflight
            .minimum_python
        )

        passed = (
            installed >= minimum
        )

        return CheckResult(
            name="Python version",

            status=(
                CheckStatus.PASS
                if passed
                else CheckStatus.FAIL
            ),

            message=(
                f"{installed} installed; "
                f">= {minimum} required."
            ),
        )


    # ========================================================
    # REQUIRED PACKAGES
    # ========================================================

    def _check_packages(
        self,
    ) -> CheckResult:

        required = {
            "torch":
                self.config
                .preflight
                .minimum_torch,

            "transformers":
                self.config
                .preflight
                .minimum_transformers,

            "peft":
                None,

            "accelerate":
                None,

            "bitsandbytes":
                None,

            "sacrebleu":
                None,

            "huggingface-hub":
                None,
        }

        details: dict[
            str,
            Any,
        ] = {}

        failures = []

        for package, minimum in required.items():

            version = package_version(
                package
            )

            details[
                package
            ] = version

            if version is None:

                failures.append(
                    f"{package} not installed"
                )

                continue

            if (
                minimum is not None
                and Version(version)
                < Version(minimum)
            ):

                failures.append(
                    f"{package} {version} "
                    f"< required {minimum}"
                )

        if failures:

            return CheckResult(
                name="Python packages",
                status=CheckStatus.FAIL,
                message="; ".join(
                    failures
                ),
                details=details,
            )

        return CheckResult(
            name="Python packages",
            status=CheckStatus.PASS,
            message=(
                "Required training packages "
                "are installed."
            ),
            details=details,
        )


    # ========================================================
    # CUDA / GPU
    # ========================================================

    def _check_cuda_gpu(
        self,
    ) -> CheckResult:

        gpu = collect_gpu_info()

        if not gpu.get(
            "cuda_available"
        ):

            status = (
                CheckStatus.FAIL
                if self.config
                .preflight
                .require_cuda
                else CheckStatus.WARN
            )

            return CheckResult(
                name="CUDA GPU",
                status=status,
                message=(
                    "CUDA GPU is not available. "
                    "In Colab select "
                    "Runtime -> Change runtime type -> GPU."
                ),
                details=gpu,
            )

        vram = float(
            gpu[
                "vram_gib"
            ]
        )

        minimum = float(
            self.config
            .preflight
            .minimum_vram_gib
        )

        if vram < minimum:

            return CheckResult(
                name="CUDA GPU",
                status=CheckStatus.FAIL,
                message=(
                    f"{gpu['gpu_name']} has "
                    f"{vram:.2f} GiB VRAM; "
                    f"our QLoRA profile expects at least "
                    f"{minimum:.2f} GiB."
                ),
                details=gpu,
            )

        return CheckResult(
            name="CUDA GPU",
            status=CheckStatus.PASS,
            message=(
                f"{gpu['gpu_name']} | "
                f"{vram:.2f} GiB | "
                f"QLoRA compute dtype: "
                f"{recommended_compute_dtype()}."
            ),
            details=gpu,
        )


    # ========================================================
    # BITSANDBYTES / NF4
    # ========================================================

    def _check_bitsandbytes_nf4(
        self,
    ) -> CheckResult:

        if not (
            self.config
            .preflight
            .check_bitsandbytes_nf4
        ):

            return CheckResult(
                name="bitsandbytes NF4",
                status=CheckStatus.WARN,
                message=(
                    "NF4 runtime test disabled "
                    "by configuration."
                ),
            )

        import torch

        if not torch.cuda.is_available():

            return CheckResult(
                name="bitsandbytes NF4",
                status=CheckStatus.FAIL,
                message=(
                    "NF4 GPU test cannot run "
                    "because CUDA is unavailable."
                ),
            )

        import bitsandbytes as bnb

        # ----------------------------------------------------
        # REAL CUDA KERNEL TEST
        # ----------------------------------------------------
        #
        # Merely importing bitsandbytes is not enough.
        #
        # We put a small FP16 tensor on the GPU and ask
        # bitsandbytes to quantize it to NF4. If the CUDA
        # backend is broken, this normally exposes it here.

        source = torch.randn(
            4096,
            device="cuda",
            dtype=torch.float16,
        )

        quantized, quant_state = (
            bnb.functional.quantize_4bit(
                source,
                quant_type="nf4",
            )
        )

        if quantized.numel() == 0:
            raise RuntimeError(
                "NF4 produced an empty tensor."
            )

        # Release this temporary allocation immediately.
        del source
        del quantized
        del quant_state

        torch.cuda.empty_cache()

        return CheckResult(
            name="bitsandbytes NF4",
            status=CheckStatus.PASS,
            message=(
                "A real NF4 quantization operation "
                "completed successfully on the GPU."
            ),
            details={
                "bitsandbytes_version":
                    package_version(
                        "bitsandbytes"
                    ),
            },
        )


    # ========================================================
    # DISK SPACE
    # ========================================================

    def _check_disk_space(
        self,
    ) -> CheckResult:

        cache_dir = Path(
            self.config
            .paths
            .hf_cache_dir
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        usage = shutil.disk_usage(
            cache_dir
        )

        free_gib = (
            usage.free
            / (1024**3)
        )

        minimum = (
            self.config
            .preflight
            .minimum_disk_free_gib
        )

        passed = (
            free_gib >= minimum
        )

        return CheckResult(
            name="Disk space",
            status=(
                CheckStatus.PASS
                if passed
                else CheckStatus.FAIL
            ),
            message=(
                f"{free_gib:.1f} GiB free near "
                f"{cache_dir}; "
                f"{minimum:.1f} GiB minimum configured."
            ),
            details={
                "cache_dir":
                    str(
                        cache_dir
                    ),

                "free_gib":
                    round(
                        free_gib,
                        3,
                    ),
            },
        )


    # ========================================================
    # ARTIFACT DIRECTORY
    # ========================================================

    def _check_artifact_writable(
        self,
    ) -> CheckResult:

        self.paths.create_directories()

        # Write and delete an actual file.
        # This catches read-only or badly mounted Drive paths.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.paths.run_root,
            prefix="write_test_",
            suffix=".tmp",
            delete=False,
        ) as handle:

            handle.write(
                "bodhan-bhili-preflight"
            )

            temporary_path = Path(
                handle.name
            )

        temporary_path.unlink(
            missing_ok=True
        )

        return CheckResult(
            name="Artifact storage",
            status=CheckStatus.PASS,
            message=(
                f"Writable: "
                f"{self.paths.run_root}"
            ),
        )


    # ========================================================
    # HUGGING FACE AUTHENTICATION
    # ========================================================

    def _get_hf_token(
        self,
    ) -> str | None:
        """Get token without ever logging or serializing it."""

        token = os.getenv(
            "HF_TOKEN"
        )

        if token:
            return token

        try:
            from huggingface_hub import (
                get_token,
            )

            return get_token()

        except Exception:
            return None


    def _check_hf_token(
        self,
    ) -> CheckResult:

        token = self._get_hf_token()

        if not token:

            return CheckResult(
                name="Hugging Face token",
                status=CheckStatus.FAIL,
                message=(
                    "HF_TOKEN was not found. "
                    "Add it to Colab Secrets; "
                    "never hard-code it in the repository."
                ),
            )

        return CheckResult(
            name="Hugging Face token",
            status=CheckStatus.PASS,
            message=(
                "Authentication token is available. "
                "The token value was not printed or saved."
            ),
        )


    # ========================================================
    # GATED BODHAN ACCESS + ARCHITECTURE
    # ========================================================

    def _check_bodhan_access(
        self,
    ) -> CheckResult:

        if not (
            self.config
            .preflight
            .check_bodhan_access
        ):

            return CheckResult(
                name="Bodhan model access",
                status=CheckStatus.WARN,
                message=(
                    "Model-access check disabled."
                ),
            )

        token = self._get_hf_token()

        if not token:

            return CheckResult(
                name="Bodhan model access",
                status=CheckStatus.FAIL,
                message=(
                    "Cannot verify gated model access "
                    "without HF_TOKEN."
                ),
            )

        from transformers import (
            AutoConfig,
        )

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # AutoConfig downloads only the small configuration
        # metadata. It does NOT download the 7.94B model weights.
        #
        # This therefore tests:
        #   - authentication;
        #   - gated-repository permission;
        #   - current Transformers understanding of Gemma 4.

        model_config = (
            AutoConfig.from_pretrained(
                self.config
                .model
                .repo_id,
                token=token,
                cache_dir=str(
                    self.config
                    .paths
                    .hf_cache_dir
                ),
            )
        )

        model_type = getattr(
            model_config,
            "model_type",
            None,
        )

        if model_type != "gemma4":

            return CheckResult(
                name="Bodhan model access",
                status=CheckStatus.FAIL,
                message=(
                    f"Model was accessible but reported "
                    f"unexpected model_type={model_type!r}; "
                    f"expected 'gemma4'."
                ),
            )

        return CheckResult(
            name="Bodhan model access",
            status=CheckStatus.PASS,
            message=(
                f"Access granted to "
                f"{self.config.model.repo_id}; "
                f"model_type=gemma4."
            ),
            details={
                "repo_id":
                    self.config
                    .model
                    .repo_id,

                "model_type":
                    model_type,

                "architectures":
                    getattr(
                        model_config,
                        "architectures",
                        None,
                    ),
            },
        )


    # ========================================================
    # chrF++ SANITY CHECK
    # ========================================================

    def _check_chrfpp(
        self,
    ) -> CheckResult:

        import sacrebleu

        # CRITICAL:
        # word_order=2 means chrF++.
        # SacreBLEU's default CHRF() is plain chrF.
        metric = sacrebleu.CHRF(
            word_order=2
        )

        # SacreBLEU needs one scoring operation before its
        # complete metric signature is available.
        metric.corpus_score(
            ["नमस्कार"],
            [["नमस्कार"]],
        )

        signature = str(
            metric.get_signature()
        )

        # SacreBLEU represents word-order 2 as nw:2.
        if "nw:2" not in signature:

            return CheckResult(
                name="chrF++ metric",
                status=CheckStatus.FAIL,
                message=(
                    "SacreBLEU signature does not "
                    "contain nw:2. We would accidentally "
                    "report chrF instead of chrF++."
                ),
                details={
                    "signature":
                        signature,
                },
            )

        return CheckResult(
            name="chrF++ metric",
            status=CheckStatus.PASS,
            message=(
                f"Verified SacreBLEU chrF++ "
                f"(word_order=2): {signature}"
            ),
        )


    # ========================================================
    # DATASET PATH
    # ========================================================

    def _check_data_path(
        self,
    ) -> CheckResult:

        path = (
            self.config
            .paths
            .raw_data_file
        )

        if path is None:

            status = (
                CheckStatus.FAIL
                if self.require_data
                else CheckStatus.WARN
            )

            return CheckResult(
                name="AIKosh dataset path",
                status=status,
                message=(
                    "No raw_data_file is configured yet. "
                    "This is acceptable for Part 1; "
                    "Part 2 will require the real AIKosh file."
                ),
            )

        path = Path(
            path
        )

        if not path.exists():

            status = (
                CheckStatus.FAIL
                if self.require_data
                else CheckStatus.WARN
            )

            return CheckResult(
                name="AIKosh dataset path",
                status=status,
                message=(
                    f"Configured file does not exist: "
                    f"{path}"
                ),
            )

        if not path.is_file():

            return CheckResult(
                name="AIKosh dataset path",
                status=CheckStatus.FAIL,
                message=(
                    f"Configured dataset path is not "
                    f"a file: {path}"
                ),
            )

        return CheckResult(
            name="AIKosh dataset path",
            status=CheckStatus.PASS,
            message=(
                f"Dataset file exists: {path}"
            ),
            details={
                "size_mib":
                    round(
                        path.stat().st_size
                        / (1024**2),
                        3,
                    ),
            },
        )


    # ========================================================
    # COMPLETE PREFLIGHT
    # ========================================================

    def run(
        self,
    ) -> dict[str, Any]:
        """Run every preflight and persist a machine-readable report."""

        checks = (
            (
                "Python version",
                self._check_python,
            ),
            (
                "Python packages",
                self._check_packages,
            ),
            (
                "CUDA GPU",
                self._check_cuda_gpu,
            ),
            (
                "bitsandbytes NF4",
                self._check_bitsandbytes_nf4,
            ),
            (
                "Disk space",
                self._check_disk_space,
            ),
            (
                "Artifact storage",
                self._check_artifact_writable,
            ),
            (
                "Hugging Face token",
                self._check_hf_token,
            ),
            (
                "Bodhan model access",
                self._check_bodhan_access,
            ),
            (
                "chrF++ metric",
                self._check_chrfpp,
            ),
            (
                "AIKosh dataset path",
                self._check_data_path,
            ),
        )

        for name, check in checks:

            self._execute(
                name,
                check,
            )

        failures = [
            result
            for result in self.results
            if result.status
            == CheckStatus.FAIL
        ]

        warnings = [
            result
            for result in self.results
            if result.status
            == CheckStatus.WARN
        ]

        report = {
            "ready":
                len(
                    failures
                )
                == 0,

            "failure_count":
                len(
                    failures
                ),

            "warning_count":
                len(
                    warnings
                ),

            "checks": [
                {
                    **asdict(
                        result
                    ),
                    "status":
                        result
                        .status
                        .value,
                }
                for result in self.results
            ],
        }

        atomic_write_json(
            report,
            self.paths.preflight_report,
        )

        return report