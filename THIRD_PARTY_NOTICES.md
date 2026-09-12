# Third-party software used by this pipeline

This repository contains no third-party SDKs. The export workflows download them from
their publishers at run time and keep them only in this repository's GitHub Actions cache.
Published model repos contain the compiled model files, never these SDKs or their runtime
libraries.

## ExecuTorch, PyTorch, torchao

Installed from PyPI (`requirements/*.txt`). ExecuTorch and PyTorch are under BSD-style
licenses; torchao under BSD-3-Clause.

## Qualcomm AI Runtime SDK (QAIRT) 2.37.0.250724

Used by `export-qnn.yml` to compile QNN HTP context binaries. The executorch 1.4.0 wheel
downloads it on first import from Qualcomm's public Software Center URL
(`https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip`).

It is licensed by Qualcomm Technologies, Inc. under the "AI Stack License" (Terms and
Conditions of Use, `LICENSE.pdf` in the SDK). By its own terms, downloading or using the
SDK is acceptance of that agreement, which every run of `export-qnn.yml` does on behalf of
whoever operates this repository. Points that bear on this pipeline, paraphrased from the
agreement (read `LICENSE.pdf` for the binding text):

- Section 1 grants a license to use and copy the SDK to develop software applications, and
  to distribute it only in object code as incorporated in such an application, never
  standalone. This pipeline does not redistribute the SDK.
- Section 2 prohibits reverse engineering, makes the user responsible for inputs and
  outputs, lists prohibited and high-risk use cases, and requires keeping copyright notices.
- Section 10.f subjects Qualcomm products and their direct products to U.S. export
  control and sanctions laws.
- The SDK's third-party notices are in `NOTICE.txt` and `QNN_NOTICE.txt` in the SDK.

Model cards of repos with `qnn/` folders state that the context binaries were compiled with
QAIRT 2.37.0.250724 and that no Qualcomm SDK or runtime library is included.

## MediaTek NeuroPilot SDK

Not used yet (docs/PLAN.md, phase 4). Its license will be reviewed and recorded here from
the text bundled in the SDK before any MediaTek export runs.
