# Kiro Web + GitHub workflow

This repository can use Kiro Web to turn GitHub issues into isolated coding-agent tasks and pull requests. This is separate from the local Kiro IDE session: GitHub-triggered work runs in Kiro Web's cloud sandbox.

## One-time connection

1. Sign in at [Kiro Web](https://app.kiro.dev) with the account that has the paid Kiro subscription.
2. Open **Settings → Integrations → GitHub → Connect GitHub**.
3. Authorize the **Kiro Agent GitHub app**.
4. Grant it access to only `mo-farooq/acoustic-reef` unless broader access is intentionally required.
5. Ensure the GitHub account used to assign work has write access to this repository.

The GitHub app must be installed and authorized for this repository, and the individual user must separately have repository access. The repository owner/account owner may need to perform the app installation or finish the setup. In this repository, the connected GitHub user `onkardamal` has push access but not repository admin access; the owner `mo-farooq` may therefore need to approve/install the app for this repository.

Official setup guide: [Kiro Web GitHub integration](https://kiro.dev/docs/web/github/).

## Assign a task from GitHub

After the app installation and your personal Kiro Web connection are complete, create a well-scoped GitHub issue with:

- a single outcome;
- context and relevant file paths;
- explicit boundaries (especially research/data restrictions);
- acceptance criteria;
- verification commands;
- a requirement to report tests and actions actually executed.

Then trigger it using either method:

- Add the `kiro` label to the issue; or
- Comment `/kiro` on the issue.

Do not add the trigger before the Kiro Agent GitHub app has been installed and the user has finished their Kiro setup. A trigger issued too early may only produce setup instructions or fail to start the task.

## What happens next

Kiro Web works in an isolated cloud sandbox, starts a feature branch from the repository's **default branch** (`main`), makes commits, pushes the branch, and opens a pull request. The local Kiro IDE is not automatically woken or controlled by this GitHub event.

Review the PR like any other code change. Never merge automatically. For feedback:

- `/kiro all` asks Kiro to address all reviewer comments on the PR.
- `/kiro fix` asks Kiro to address a specific review conversation.

Only the user who started the Kiro task can direct that task through GitHub feedback commands.

## First-task safety checklist

Before assigning the first substantive issue:

- Merge the documentation/steering PR that adds `AGENTS.md` and `.kiro/steering/`. Kiro Web starts from `main`, so it will not see steering files that still exist only on an unmerged feature branch.
- Confirm the repository appears in Kiro Web after GitHub connection.
- Start with a bounded review or test-only task; do not ask the agent to download audio, extract features, train models, or run statistical experiments.
- Require a pull request and human review; do not enable auto-merge.

## Repository-specific constraints

The shared agent guardrails are in [`AGENTS.md`](../AGENTS.md) and [`.kiro/steering/`](../.kiro/steering/). They preserve the prototype/research boundary and the current E0/E1 status. Read [research status](research-status.md), [data provenance](data-provenance.md), [architecture and scope](architecture-and-scope.md), and [prototype audit](prototype-audit.md) before technical work.
