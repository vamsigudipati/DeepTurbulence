# DeepTurbulence Guide Hub

This folder contains four coordinated documents derived from mining the DeepTurbulence source tree using the topology-first workflow:

| File | Audience | Contents |
| --- | --- | --- |
| [topology.md](./topology.md) | Everyone (start here) | Structural map (Mermaid) + entity inventory + dependency edges. The backbone for the other docs. |
| [developer_guide.md](./developer_guide.md) | Contributors / framework developers | Architecture, module tours, extension SOPs, debugging tips. |
| [user_guide.md](./user_guide.md) | End users / applied engineers | Installation, configuration, and how to run the training/prediction scripts. |
| [tutorial.md](./tutorial.md) | Learners at any level | 10-chapter "from beginner to expert" walkthrough, grounded in the MATLAB and Python scripts in this repository. |

> Upstream project: [DeepTurbulence](https://github.com/lguas/Deepturb)
> Canonical citation: P. A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa, "Predictions of turbulent shear flows using deep neural networks", *Phys. Rev. Fluids* **4**, 054603 (2019).

## How these documents were produced

- Based on direct reading of the [Data generator (Moehlis model)/](../../Data%20generator%20%28Moehlis%20model%29) MATLAB scripts, the [Neural networks models/](../../Neural%20networks%20models) Python/Keras scripts, and the reference paper in [docs/references/](../references/).
- Every claim is grounded in a concrete source path; unresolved facts are marked with `> TODO(doc-miner): ...`.

## Reading order suggestions

1. **New users** — start with `user_guide.md` §1–§11, then do chapters 1–4 of `tutorial.md`.
2. **Applied engineers** with a specific task — jump to the matching chapter of `tutorial.md`.
3. **Contributors** — read `developer_guide.md` end to end; cross-reference `user_guide.md` when a user-facing workflow is discussed.
