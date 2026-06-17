# Paulitools Demos

## Pauli Branching Demo

`pauli_branching_demo.py` showcases how to:

1. Generate random Pauli strings on a fixed number of qubits.
2. Persist each generated Pauli to a `.ptstore` archive using `paulitools.save_pauli_data`.
3. Track commutation and anti-commutation relationships as the collection grows.
4. Maintain a running row-reduced basis with `paulitools.row_reduce`.
5. Summarise anti-commuting clusters and their cardinalities after each insertion.

### Running the demo

From the project root:

```bash
python -m demos.pauli_branching_demo
```

You can customise the number of qubits, rounds, or storage path:

```bash
python -m demos.pauli_branching_demo 4 12 pauli_data.ptstore
```

The script prints each step’s Pauli, the updated row-reduced basis, and the histogram of anti-commuting cluster sizes. The `.ptstore` archive is overwritten on each run.
