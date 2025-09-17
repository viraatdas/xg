#!/bin/bash

set -e

echo "=== XG Language CLI Examples ==="
echo

mkdir -p build

echo "=== 1. Compiling cluster_matmul.xg for H100 ==="
xgc examples/cluster_matmul.xg --target=H100 --num-gpu=4 --verbose --out build/cluster_h100.xge
echo

echo "=== 2. Compiling for different targets ==="
xgc examples/cluster_matmul.xg --target=GB200 --num-gpu=8 --verbose --out build/cluster_gb200.xge
xgc examples/cluster_matmul.xg --target=A100 --num-gpu=2 --verbose --out build/cluster_a100.xge
echo

echo "=== 3. Creating external values for testing ==="
cat > build/values.json << EOF
{
  "A": {
    "tensor": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    "dtype": "float32"
  },
  "B": {
    "tensor": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    "dtype": "float32"
  }
}
EOF
echo "Created build/values.json"
echo

echo "=== 4. Running compiled engine ==="
xgrun build/cluster_h100.xge --external build/values.json --verbose --profile
echo

echo "=== 5. Running source directly ==="
xgrun examples/cluster_matmul.xg --external build/values.json --verbose --profile
echo

echo "=== 6. Testing advanced example ==="
cat > build/advanced_values.json << EOF
{
  "A": {
    "tensor": [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]
    ],
    "dtype": "float32"
  },
  "B": {
    "tensor": [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]
    ],
    "dtype": "float32"
  }
}
EOF

xgrun examples/advanced_cluster_matmul.xg --external build/advanced_values.json --verbose --profile
echo

echo "=== 7. Compilation check only ==="
xgc examples/cluster_matmul.xg --check-only --verbose
xgc examples/advanced_cluster_matmul.xg --check-only --verbose
echo

echo "=== 8. Testing different matrix sizes ==="
cat > build/large_values.json << EOF
{
  "A": {
    "tensor": [
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
      [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
    ],
    "dtype": "float32"
  },
  "B": {
    "tensor": [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
      [17.0, 18.0, 19.0, 20.0],
      [21.0, 22.0, 23.0, 24.0],
      [25.0, 26.0, 27.0, 28.0],
      [29.0, 30.0, 31.0, 32.0]
    ],
    "dtype": "float32"
  }
}
EOF

xgrun examples/cluster_matmul.xg --external build/large_values.json --verbose
echo

echo "=== All examples completed successfully! ==="
echo "Generated files:"
ls -la build/
