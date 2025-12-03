use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

#[pyclass]
struct RustDataLoader {
    bin_mmap: Arc<Mmap>,
    indices: Vec<(u64, u64)>, // offset, length
    rng: ThreadRng,
}

#[pymethods]
impl RustDataLoader {
    #[new]
    fn new(path: String, split: String) -> PyResult<Self> {
        let root = Path::new(&path);
        let bin_path = root.join(format!("{}.bin", split));
        let idx_path = root.join(format!("{}.idx", split));

        if !bin_path.exists() || !idx_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Missing bin/idx files under {} for split '{}'",
                path, split
            )));
        }

        // Read Index
        let mut idx_file = File::open(&idx_path)?;
        let mut magic = [0u8; 4];
        idx_file.read_exact(&mut magic)?;
        if &magic != b"MUSE" {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid magic in idx file"));
        }
        let _version = idx_file.read_u32::<LittleEndian>()?;
        
        // Read indices (u64, u64)
        let mut indices = Vec::new();
        loop {
            match idx_file.read_u64::<LittleEndian>() {
                Ok(offset) => {
                    let length = idx_file.read_u64::<LittleEndian>()?;
                    indices.push((offset, length));
                }
                Err(_) => break, // EOF
            }
        }

        // Mmap Bin
        let bin_file = File::open(&bin_path)?;
        let mmap = unsafe { Mmap::map(&bin_file)? };

        Ok(RustDataLoader {
            bin_mmap: Arc::new(mmap),
            indices,
            rng: rand::thread_rng(),
        })
    }

    fn num_docs(&self) -> usize {
        self.indices.len()
    }

    fn sample_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
        seq_len: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
        // Parallel sampling using Rayon?
        // Rayon needs Send/Sync. ThreadRng is not Send.
        // We can use thread_local rngs or create rngs per thread.
        
        // For simplicity and to avoid GIL issues with parallel allocation,
        // let's try sequential first, or parallel data fetch then copy.
        
        let mut x_batch = Vec::with_capacity(batch_size * seq_len);
        let mut y_batch = Vec::with_capacity(batch_size * seq_len);
        
        // We can parallelize the sampling logic
        let indices = &self.indices;
        let mmap = &self.bin_mmap;
        
        // Create a parallel iterator
        let samples: Vec<(Vec<i64>, Vec<i64>)> = (0..batch_size).into_par_iter().map(|_| {
            let mut rng = rand::thread_rng();
            loop {
                let doc_idx = rng.gen_range(0..indices.len());
                let (offset_bytes, length) = indices[doc_idx];
                // offset in bytes. length in tokens (uint32).
                // Wait, python code says:
                // idx_data = np.fromfile(f, dtype=np.uint64)
                // self.index = idx_data.reshape(-1, 2)
                // offset, length = self.index[doc_id]
                // slice_tokens = self.tokens[offset + start : offset + end]
                // self.tokens is uint32 memmap.
                // So offset in .idx is likely in TOKENS (uint32 index), not bytes?
                // Or bytes?
                // Usually np.fromfile reads raw bytes.
                // But prepare_datasets.py defines what it writes.
                // Assuming offset is in TOKENS (element index) because numpy indexing uses element index.
                // If it was bytes, numpy would need byte offset? No, numpy memmap uses offset in bytes for start, but indexing is elements.
                // Let's assume offset is in TOKENS.
                
                let offset = offset_bytes as usize; // offset in tokens
                let len = length as usize;
                
                if len <= seq_len {
                    continue;
                }
                
                let start = rng.gen_range(0..len - seq_len);
                let end = start + seq_len + 1;
                
                // Read from mmap
                // mmap is bytes. We need to read u32.
                // offset * 4
                let byte_start = (offset + start) * 4;
                let byte_end = (offset + end) * 4;
                
                if byte_end > mmap.len() {
                    continue; // Safety
                }
                
                let slice = &mmap[byte_start..byte_end];
                
                // Convert to i64 (for torch LongTensor)
                // slice is u8. We need to interpret as u32 then cast to i64.
                let mut tokens = Vec::with_capacity(seq_len + 1);
                for chunk in slice.chunks_exact(4) {
                    let val = u32::from_le_bytes(chunk.try_into().unwrap());
                    tokens.push(val as i64);
                }
                
                let x = tokens[0..seq_len].to_vec();
                let y = tokens[1..seq_len+1].to_vec();
                
                return (x, y);
            }
        }).collect();
        
        // Flatten
        for (x, y) in samples {
            x_batch.extend(x);
            y_batch.extend(y);
        }
        
        let x_np = PyArray2::from_vec(py, [batch_size, seq_len], x_batch);
        let y_np = PyArray1::from_vec(py, y_batch); // Flattened targets
        
        Ok((x_np, y_np))
    }

    fn sample_sequence<'py>(
        &mut self,
        py: Python<'py>,
        seq_len: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>> {
        // Single sequence sampling
        let indices = &self.indices;
        let mmap = &self.bin_mmap;
        
        for _ in 0..8 {
            let doc_idx = self.rng.gen_range(0..indices.len());
            let (offset_bytes, length) = indices[doc_idx];
            let offset = offset_bytes as usize;
            let len = length as usize;
            
            if len <= seq_len {
                continue;
            }
            
            let start = self.rng.gen_range(0..len - seq_len);
            let end = start + seq_len + 1;
            
            let byte_start = (offset + start) * 4;
            let byte_end = (offset + end) * 4;
            
            if byte_end > mmap.len() {
                continue;
            }
            
            let slice = &mmap[byte_start..byte_end];
            
            let mut tokens = Vec::with_capacity(seq_len + 1);
            for chunk in slice.chunks_exact(4) {
                let val = u32::from_le_bytes(chunk.try_into().unwrap());
                tokens.push(val as i64);
            }
            
            let x = PyArray1::from_vec(py, tokens[0..seq_len].to_vec());
            let y = PyArray1::from_vec(py, tokens[1..seq_len+1].to_vec());
            
            return Ok(Some((x, y)));
        }
        
        Ok(None)
    }
}

#[pymodule]
fn rust_loader(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustDataLoader>()?;
    Ok(())
}
