# Ingestion Pipeline Experiments

This directory contains experiments related to document ingestion and processing pipelines.

## Current Experiments

### `ing_pip_test.ipynb` - Chunk Size Optimization
- **Objective**: Test different chunk sizes for document ingestion
- **Current Status**: Testing 400 vs 200 token chunks
- **Metrics**: Number of chunks, processing time, embedding quality
- **Dataset**: Calculus textbook, Physics contract

### Next Experiments
- [ ] Different chunking strategies (recursive, sentence-based, semantic)
- [ ] Embedding model comparison
- [ ] Database optimization
- [ ] Parallel processing evaluation

## Running Experiments

1. **Setup environment**:
   ```bash
   cd research/experiments/ingestion/
   jupyter notebook
   ```

2. **Configure database**:
   Make sure your `.env` file contains:
   ```
   DB_HOST=your_host
   DB_USER=your_user
   DB_PASSWORD=your_password  
   DB_DATABASE=your_database
   ```

3. **Run notebook**:
   Open `ing_pip_test.ipynb` and run cells sequentially

## Results

Results are automatically saved to:
- **Database**: Embeddings stored in PostgreSQL
- **Logs**: Console output for debugging
- **Metrics**: Can be exported to `../../../outputs/experiments_results/`
