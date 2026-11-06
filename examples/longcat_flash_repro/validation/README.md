# Validation scope

The branch workflow reuses the reviewed experimental templates. Dependency audit is read-only and lists all missing or mismatched reference packages before framework imports. No installer, dependency archive, virtual environment, native binary or checkpoint is delivered through Git.

CPU tests cover publication scans, model-baseline descendants, explicit device lists, relocation with spaces and quotes, generated-cache exclusion, x86/ARM ELF checks, available device counts, and source checksum verification. Original component and Trainer measurements were reaggregated without changing numerical or performance verdicts.

Actual x86 NPU end-to-end validation remains a customer-side step. Every run executes the numerical phase before accepting performance results.
