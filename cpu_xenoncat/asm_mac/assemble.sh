./fasm -m 1280000 ../asm_linux/equihash_avx1.asm equihash_avx1.elf.o
./fasm -m 1280000 ../asm_linux/equihash_avx2.asm equihash_avx2.elf.o
./objconv -fmacho64 -nu equihash_avx1.elf.o equihash_avx1.o
./objconv -fmacho64 -nu equihash_avx2.elf.o equihash_avx2.o
