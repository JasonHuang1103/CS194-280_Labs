import os
from src.embedding_db import VectorDB
from src.embedding_models import MiniEmbeddingModel
import time
import argparse
import numpy as np
import pickle
import sys

# Create the documents directory if it doesn't exist
os.makedirs("documents", exist_ok=True)

# List of URLs to scrape
urls = [
    "https://lean-lang.org/theorem_proving_in_lean4/propositions_and_proofs.html",
    "https://lean-lang.org/theorem_proving_in_lean4/quantifiers_and_equality.html",
    "https://lean-lang.org/theorem_proving_in_lean4/tactics.html",
    "https://lean-lang.org/theorem_proving_in_lean4/interacting_with_lean.html",
    "https://lean-lang.org/theorem_proving_in_lean4/type_classes.html",
    "https://lean-lang.org/theorem_proving_in_lean4/axioms_and_computation.html",
    "https://lean-lang.org/functional_programming_in_lean/type-classes/indexing.html",
    "https://lean-lang.org/functional_programming_in_lean/programs-proofs/inequalities.html",
    "https://lean-lang.org/doc/reference/latest//Basic-Types/Natural-Numbers/#Nat",
    "https://lean-lang.org/doc/reference/latest//Basic-Types/Integers/#Int",
    "https://lean-lang.org/doc/reference/latest//Basic-Types/Arrays/#Array",
    "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://pp.ipd.kit.edu/uploads/publikationen/demoura21lean4.pdf",
    "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2407.03203",
    "https://github.com/leanprover/lean4/blob/master/src/Init/Data/Array/Basic.lean",
    "https://github.com/leanprover/lean4/blob/master/src/Init/Core.lean",
    "https://proofassistants.stackexchange.com/questions/2071/using-the-contrapositive-in-lean-4",
    "https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html",
    "https://learnxinyminutes.com/lean4/",
    "https://proofassistants.stackexchange.com/questions/4297/lean-4-function-that-switches-on-proposition",
    "https://proofassistants.stackexchange.com/questions/1797/getting-an-element-of-a-list-in-lean",
    "https://proofassistants.stackexchange.com/questions/1888/tactics-for-array-list-simplification-in-lean4",
    "https://proofassistants.stackexchange.com/questions/4113/how-to-perform-type-conversion-coercion-in-lean-4",
    "https://proofassistants.stackexchange.com/questions/1380/how-do-i-convince-the-lean-4-type-checker-that-addition-is-commutative",
    "https://leanprover-community.github.io/theories/naturals.html",
]

def get_embedding_model():
    """
    Gets the MiniEmbeddingModel for database operations.
    
    Returns:
        An instance of MiniEmbeddingModel
    """
    print("Using MiniLM embedding model")
    return MiniEmbeddingModel()

def scrape_all():
    """
    Scrapes all URLs in the urls list and saves them to the documents directory.
    """
    print(f"Starting to scrape {len(urls)} Lean documentation pages...")
    
    for i, url in enumerate(urls):
        # Extract a filename from the URL
        filename = os.path.basename(url).replace(".html", "")
        output_file = os.path.join("documents", f"{filename}.txt")
        
        print(f"[{i+1}/{len(urls)}] Scraping {url} to {output_file}")
        
        try:
            # Use VectorDB's static method to scrape the website
            VectorDB.scrape_website(url, output_file)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)
    
    print("\nScraping completed successfully!")

def build_database():
    """
    Builds the vector database from the scraped documents using MiniEmbeddingModel.
    """
    print("\nBuilding vector database from the scraped documents...")
    
    # Initialize the embedding model
    embedding_model = get_embedding_model()
    
    try:
        # Create and save the database
        VectorDB(
            directory="documents",
            vector_file="database.npy",
            embedding_model=embedding_model
        )
        print("Vector database built successfully!")
    except Exception as e:
        print(f"Error building database: {e}")
        sys.exit(1)

def process_chunks_in_batches(embedding_model, chunks, batch_size=10):
    """
    Process chunks in batches to avoid timeouts.
    
    Args:
        embedding_model: The embedding model to use
        chunks: List of text chunks to process
        batch_size: Size of each batch
        
    Returns:
        Numpy array of embeddings
    """
    print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
        
        try:
            batch_embeddings = embedding_model.get_embeddings_batch(batch)
            all_embeddings.append(batch_embeddings)
            # Short pause between batches
            time.sleep(1)
        except Exception as e:
            print(f"Error processing batch: {e}")
            print("Skipping problematic batch and continuing...")
            continue
    
    if not all_embeddings:
        raise ValueError("Failed to generate any embeddings")
    
    # Combine all batch results
    return np.vstack(all_embeddings)

def append_all_to_database():
    """
    Scrapes all URLs in the list and appends them to the existing database.
    Creates a new database if one doesn't exist.
    """
    database_file = "database.npy"
    chunks_file = "database_chunks.pkl"
    
    # Check if database exists
    if not os.path.exists(database_file) or not os.path.exists(chunks_file):
        print("Database doesn't exist. Will create a new one by scraping all URLs...")
        scrape_all()
        build_database()
        return
    
    print(f"Appending {len(urls)} URLs to the existing database...")
    
    # Initialize the embedding model
    embedding_model = get_embedding_model()
    
    try:
        # Load the existing embeddings and chunks
        existing_embeddings = np.load(database_file)
        with open(chunks_file, 'rb') as f:
            existing_chunks = pickle.load(f)
        
        print(f"Current database has {len(existing_chunks)} chunks with dimension {existing_embeddings.shape[1]}")
        
        # Process each URL
        new_docs = []
        successful_urls = []
        
        for i, url in enumerate(urls):
            # Extract a filename from the URL
            filename = os.path.basename(url).replace(".html", "")
            output_file = os.path.join("documents", f"{filename}.txt")
            
            print(f"[{i+1}/{len(urls)}] Scraping {url} to {output_file}")
            
            try:
                # Use VectorDB's static method to scrape the website
                VectorDB.scrape_website(url, output_file)
                
                # Read the new document
                with open(output_file, 'r', encoding='utf-8') as f:
                    new_docs.append(f.read())
                
                successful_urls.append(url)
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(1)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        if not new_docs:
            print("No documents were successfully scraped. Exiting.")
            return
        
        print(f"Successfully scraped {len(new_docs)} documents. Processing chunks and embeddings...")
        
        # Split all new documents into chunks
        new_chunks = embedding_model.split_documents(new_docs)
        
        # Get embeddings for the new chunks in batches
        new_embeddings = process_chunks_in_batches(embedding_model, new_chunks)
        
        # Verify dimensions match
        if existing_embeddings.shape[1] != new_embeddings.shape[1]:
            print(f"ERROR: Dimension mismatch! Existing: {existing_embeddings.shape[1]}, New: {new_embeddings.shape[1]}")
            print("This means you're trying to use a different model than what was used to create the database.")
            print("Please rebuild the database with --rebuild first.")
            return
        
        # Combine the old and new data
        combined_chunks = existing_chunks + new_chunks
        combined_embeddings = np.vstack((existing_embeddings, new_embeddings))
        
        # Save the updated database
        np.save(database_file, combined_embeddings)
        with open(chunks_file, 'wb') as f:
            pickle.dump(combined_chunks, f)
        
        print(f"Database updated successfully. Now contains {len(combined_chunks)} chunks.")
    except Exception as e:
        print(f"Error appending to database: {e}")
        sys.exit(1)

def rebuild_database():
    """
    Rebuilds the database using existing document files with MiniEmbeddingModel.
    """
    print("Rebuilding database with MiniLM model")
    
    # Remove existing database files
    database_file = "database.npy"
    chunks_file = "database_chunks.pkl"
    
    if os.path.exists(database_file):
        os.remove(database_file)
        print(f"Removed existing database file: {database_file}")
    
    if os.path.exists(chunks_file):
        os.remove(chunks_file)
        print(f"Removed existing chunks file: {chunks_file}")
    
    # Build fresh database
    build_database()

def test_query(query="How do I write a proof in Lean?"):
    """
    Tests a query against the database.
    
    Args:
        query: The query to test
    """
    database_file = "database.npy"
    
    if not os.path.exists(database_file):
        print("Error: Database file not found. Please build the database first.")
        return
    
    print(f"\nTesting query: '{query}'")
    embedding_model = get_embedding_model()
    
    try:
        top_k_chunks, top_k_scores = VectorDB.get_top_k(
            "database.npy", 
            embedding_model, 
            query, 
            k=3, 
            verbose=True
        )
        print("Database is ready to use for semantic search in your Lean proofs!")
    except Exception as e:
        print(f"Error querying the database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Lean documentation and build a vector database")
    parser.add_argument("--append-all", action="store_true", help="Append all URLs in the list to the existing database")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild database (keeps documents)")
    parser.add_argument("--test", action="store_true", help="Test the database with a sample query after operation")
    parser.add_argument("--query", type=str, default="How do I write a proof in Lean?", help="Query to test against the database")
    
    args = parser.parse_args()
    
    if not args.append_all and not args.rebuild:
        # Default to append-all if no arguments specified
        args.append_all = True
        print("No options specified. Defaulting to --append-all")
    
    # Handle rebuild option
    if args.rebuild:
        rebuild_database()
    
    # Handle append-all option
    if args.append_all:
        append_all_to_database()
    
    # Test the database if requested
    if args.test:
        test_query(args.query)
    
    print("Scraper operations completed successfully!")
