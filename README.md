# FP-Growth for FIMI Data

This project implements the FP-Growth algorithm in pure Python for mining frequent itemsets and association rules from transaction data in FIMI format.

## Features
- Reads transaction data in FIMI format (one transaction per line, items separated by whitespace)
- Mines frequent itemsets using the FP-Growth algorithm
- Filters maximal itemsets
- Generates association rules with support for confidence, lift, leverage, and conviction metrics

## File Structure
- `mine_fimi.py`: Main script containing the FP-Growth implementation and utilities
- `retail.dat`: Example dataset in FIMI format

## Usage

### Requirements
- Python 3.x (no external dependencies required)

### Running the Script

```bash
python mine_fimi.py --input retail.dat --minsup 0.01
```

#### Arguments
- `--input`: Path to the FIMI-format transaction file
- `--minsup`: Minimum support threshold (as a fraction, e.g., 0.01 for 1%)
- Additional options may be available for filtering maximal itemsets or generating association rules (see script for details)

## FIMI Format
Each line in the input file represents a transaction. Items are separated by whitespace. Example:

```
1 2 3
2 3 4
1 2
```

## License
This project is for educational and research purposes.

