#/bin/sh
POLYBENCH_DIR=$1
OUTPUT_DIR="polybench_so_files"
if [ -z "$POLYBENCH_DIR" ]; then
    echo "Usage: $0 <polybench_directory>"
    exit 1
fi
mkdir Polybench_so_files
for benchmark in $(find "$POLYBENCH_DIR" \
                    -type f -name "*.c" \
                    -not -name "*Nussinov.orig*" \
                    -not -name "*polybench*" \
                    -not -name "*template*"); do
    benchmark_name=$(basename "$benchmark" .c)
    benchmark_dir=$(dirname "$benchmark")
    sed ':a;N;$!ba;s/static\n\(\s*void kernel\)/\1/g' "$benchmark" > "${benchmark_name}_mod.c"
    cp "$benchmark_dir/$benchmark_name.h" "$benchmark_name.h"
    gcc -DPOLYBENCH_TIME \
        -DDATA_TYPE_IS_FLOAT \
        $POLYBENCH_DIR/utilities/polybench.c \
        -shared -DSMALL_DATASET \
        -I$POLYBENCH_DIR/utilities/ \
        -fPIC -lm -DPOLYBENCH_DUMP_ARRAYS \
        -O3 ${benchmark_name}_mod.c \
        -o "$benchmark_name.so"
    cp "$benchmark_name.so" "$OUTPUT_DIR/$benchmark_name.so"
    rm "$benchmark_name.so"
    rm "${benchmark_name}_mod.c"
    rm "${benchmark_name}.h"
    echo "Generated file: ${OUTPUT_DIR}/${benchmark_name}.so"
done
