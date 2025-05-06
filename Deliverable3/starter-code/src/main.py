import os
from src.agents import Reasoning_Agent, LLM_Agent
from src.lean_runner import execute_lean_code
from typing import Dict, List, Tuple, TypeAlias
import re
import datetime
import json
import sys
import numpy as np

# Define type alias for Lean code
LeanCode: TypeAlias = Dict[str, str]

# Flag to determine if RAG components are available
RAG_AVAILABLE = False
RAG_SETUP_SUCCESSFUL = False
try:
    from src.embedding_db import VectorDB
    from src.embedding_models import MiniEmbeddingModel
    RAG_AVAILABLE = True
    print("RAG components successfully imported.")
except ImportError as e:
    print(f"ERROR: RAG functionality is not available due to import errors: {e}")
    print("Exiting. Please install all necessary RAG components.")
    sys.exit(1)

# List of task IDs matching the one in tests.py
task_ids = [0, 58, 77, 127, 227, 404, 431, 433, 435, 441, 447]
# task_ids = [227, 404, 431, 433, 447]

# Configure which task IDs use RAG
# True: Use RAG with MiniEmbeddingModel
# False: Don't use RAG
RAG_CONFIG = {
    0: False,    # No RAG
    58: False,   # No RAG
    77: False,   # No RAG
    127: False,  # No RAG
    227: True,   # Use MiniEmbedding
    404: True,   # Use MiniEmbedding
    431: True,   # Use MiniEmbedding
    433: True,   # Use MiniEmbedding
    435: True,   # Use MiniEmbedding
    441: False,  # No RAG
    447: True,   # Use MiniEmbedding
}

# Global timestamp for the current run (created once at import time)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Global counter to track which task we're processing
CURRENT_TASK_INDEX = 0

# Global counter to track file sequence for each task
RESPONSE_COUNTERS = {}

def detect_db_model_type(database_file):
    """
    Detects which model type was used to create the database by checking the embedding dimensions.
    
    Args:
        database_file: Path to the database file
        
    Returns:
        String indicating the model type ('openai' or 'mini')
    """
    try:
        embeddings = np.load(database_file)
        if embeddings.shape[1] == 1536:
            print(f"Detected OpenAI embedding model (1536 dimensions)")
            return 'openai'
        elif embeddings.shape[1] == 384:
            print(f"Detected MiniLM embedding model (384 dimensions)")
            return 'mini'
        else:
            print(f"WARNING: Unknown embedding dimension: {embeddings.shape[1]}")
            return 'unknown'
    except Exception as e:
        print(f"Error detecting model type: {e}")
        return 'unknown'

def get_embedding_model_for_database(database_file=None, task_id=None):
    """
    Gets the appropriate embedding model based on task_id.
    
    Args:
        database_file: Path to the database file (not used)
        task_id: The task ID (optional)
        
    Returns:
        An instance of MiniEmbeddingModel or None if RAG should not be used
    """
    # Extract the numeric task ID if given in string format
    if isinstance(task_id, str) and task_id.startswith('task_id_'):
        try:
            task_id = int(task_id.replace('task_id_', ''))
        except ValueError:
            pass
    
    # Check if we have a specific configuration for this task
    if task_id is not None and task_id in RAG_CONFIG:
        use_rag = RAG_CONFIG[task_id]
        
        # Return None if we should not use RAG for this task
        if not use_rag:
            print(f"RAG is disabled for task {task_id}")
            return None
    
    # Return MiniEmbeddingModel in all other cases
    print("Using MiniLM embedding model")
    return MiniEmbeddingModel()

def save_response(task_id: str, response: str, agent_type: str) -> str:
    """
    Saves an agent's response to a file in a directory structure:
    results/timestamp/task_id/counter_agent_type.txt
    
    Args:
        task_id: The ID of the task (e.g., "task_id_0")
        response: The response content to save
        agent_type: Type of agent response (e.g., "planning", "code", "proof", "verification")
    
    Returns:
        The path to the saved file
    """
    global RESPONSE_COUNTERS
    
    # Initialize counter for this task if not already present
    if task_id not in RESPONSE_COUNTERS:
        RESPONSE_COUNTERS[task_id] = 1
    
    # Get current counter value for this task
    counter = RESPONSE_COUNTERS[task_id]
    
    # Create main results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp directory for this run
    timestamp_dir = os.path.join(results_dir, RUN_TIMESTAMP)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    # Create task-specific subdirectory
    task_dir = os.path.join(timestamp_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Create filename with counter and agent_type
    filename = f"{counter:02d}_{agent_type}.txt"
    file_path = os.path.join(task_dir, filename)
    
    # Increment the counter for next file
    RESPONSE_COUNTERS[task_id] += 1
    
    # Save the response to the file
    with open(file_path, "w") as f:
        f.write(response)
    
    print(f"Response saved to {file_path}")
    return file_path

def main_workflow(problem_description: str, task_lean_code: str = "") -> LeanCode:
    """
    Main workflow for the coding agent. This workflow takes in the problem description in natural language (description.txt) 
    and the corresponding Lean code template (task.lean). It returns the function implementation and the proof in Lean.
    
    Args:
        problem_description: Problem description in natural language. This file is read from "description.txt"
        task_lean_code: Lean code template. This file is read from "task.lean"
    
    Returns:
        LeanCode: Final generated solution, which is a dictionary with two keys: "code" and "proof".
    """
    global CURRENT_TASK_INDEX, RAG_SETUP_SUCCESSFUL
    
    # Get the task_id from the global counter
    if CURRENT_TASK_INDEX < len(task_ids):
        task_id = f"task_id_{task_ids[CURRENT_TASK_INDEX]}"
        current_task_num = task_ids[CURRENT_TASK_INDEX]
        # Increment for next call
        CURRENT_TASK_INDEX += 1
    else:
        # If we've gone through all tasks, reset and start over
        CURRENT_TASK_INDEX = 0
        task_id = f"task_id_{task_ids[CURRENT_TASK_INDEX]}"
        current_task_num = task_ids[CURRENT_TASK_INDEX]
        CURRENT_TASK_INDEX += 1
    
    print(f"Starting workflow run with timestamp: {RUN_TIMESTAMP}")
    print(f"Processing task: {task_id} (Task index: {CURRENT_TASK_INDEX-1})")
    
    # Get the function name from the template
    function_name, spec_def, parameters = extract_info_from_template(task_lean_code)
    
    # Initialize three agents for our workflow
    planning_agent = LLM_Agent(model="o3-mini")
    generation_agent = LLM_Agent(model="gpt-4o")
    verification_agent = Reasoning_Agent(model="o3-mini")
    
    # Initialize RAG retrieval
    relevant_examples = []
    proof_examples = []
    
    # Check if we should use RAG for this task
    use_rag = RAG_CONFIG.get(current_task_num, False)  # Default to False if not specified
    
    if not use_rag:
        print(f"RAG is disabled for task {current_task_num}")
        RAG_SETUP_SUCCESSFUL = False
        save_response(task_id, "RAG disabled for this task", "rag_status")
    else:
        # RAG setup is enabled for this task
        try:
            embedding_database_file = "database.npy"
            
            # Create embedding database if it doesn't exist
            if not os.path.exists(embedding_database_file):
                print(f"ERROR: Embedding database file '{embedding_database_file}' not found.")
                print("Please run the scraper.py script to create the database first.")
                sys.exit(1)
            
            # Get the appropriate embedding model for the task (always MiniEmbeddingModel)
            embedding_model = get_embedding_model_for_database(task_id=current_task_num)
            
            if embedding_model is None:
                # RAG is disabled for this task
                RAG_SETUP_SUCCESSFUL = False
                save_response(task_id, "RAG disabled for this task", "rag_status")
            else:
                # Retrieve relevant examples for this task
                print(f"Retrieving relevant examples for function '{function_name}'...")
                query = f"Lean 4 {function_name} implementation and proof example with {spec_def}"
                
                try:
                    top_chunks, _ = VectorDB.get_top_k(embedding_database_file, embedding_model, query, k=3)
                    relevant_examples = top_chunks
                    save_response(task_id, "\n".join(relevant_examples), "rag_examples")
                    
                    # Try to get proof-specific examples if available
                    print(f"Retrieving proof examples...")
                    proof_query = f"Lean 4 proof examples for {function_name} with specification {spec_def}"
                    proof_chunks, _ = VectorDB.get_top_k(embedding_database_file, embedding_model, proof_query, k=3)
                    if proof_chunks:
                        proof_examples = proof_chunks
                        save_response(task_id, "\n".join(proof_examples), "proof_rag_examples")
                    else:
                        # Fall back to general examples if no proof-specific ones found
                        proof_examples = relevant_examples
                    
                    RAG_SETUP_SUCCESSFUL = True
                except Exception as e:
                    print(f"ERROR: RAG query failed: {e}")
                    print("Please check your database and embedding model setup.")
                    sys.exit(1)
        except Exception as e:
            print(f"ERROR: RAG setup failed: {e}")
            print("Please check your database and embedding model setup.")
            sys.exit(1)
    
    if not RAG_SETUP_SUCCESSFUL and use_rag:
        print("ERROR: RAG setup was not successful. Exiting.")
        sys.exit(1)
    
    # Step 1: Planning - the planning agent will analyze the problem and create a strategy
    planning_template = """
        I need to implement a Lean 4 function and proof. Here's the problem description:

        {problem_description}

        Here's the Lean template:

        {task_lean_code}
        
        {examples}

        Please analyze this problem step-by-step:
        1. First, understand what the function should do based on the problem description and template
        2. Break down the problem into smaller subproblems
        3. Consider the input types, edge cases, and expected behavior
        4. Think about what proof techniques would be appropriate
        5. Note any similar patterns from the examples that might help

        For each step of your thinking, explicitly walk through your reasoning process.
        Conclude with a detailed plan for both implementation and proof.
    """
    
    # Add relevant examples if available
    planning_examples_text = ""
    if relevant_examples:
        planning_examples_text = "Here are some relevant examples from the Lean 4 documentation and examples that might help:\n\n" + "\n\n".join(relevant_examples)
    
    # Format the planning content
    formatted_planning_content = planning_template.format(
        problem_description=problem_description,
        task_lean_code=task_lean_code,
        examples=planning_examples_text
    )
    
    planning_prompt = [
        {"role": "system", "content": "You are a Lean 4 expert and theorem prover. Your task is to develop a plan for implementing a function and proving it meets the specifications."},
        {"role": "user", "content": formatted_planning_content}
    ]
    
    plan = planning_agent.get_response(planning_prompt)
    save_response(task_id, plan, "planning")
    
    # Step 2: Generate initial code implementation
    user_content = """
        I need to implement a Lean 4 function. Here's the problem description:

        {problem_description}

        Here's the Lean template with placeholders:

        {task_lean_code}

        Planning strategy:
        {plan}
        
        {examples}

        Please implement only the function definition that should replace the {{code}} placeholder. Do not include the 'def {function_name}' part, comments, or any other template code. Just provide the function body.
    """
    
    # Add relevant examples if available
    examples_text = ""
    if relevant_examples:
        examples_text = "Here are some relevant examples that might help:\n\n" + "\n\n".join(relevant_examples)
    
    # Format the content with proper values
    formatted_user_content = user_content.format(
        problem_description=problem_description,
        task_lean_code=task_lean_code,
        plan=plan,
        examples=examples_text,
        function_name=function_name
    )
    
    code_gen_prompt = [
        {"role": "system", "content": "You are a Lean 4 expert programmer. Your task is to generate code based on a problem description and template."},
        {"role": "user", "content": formatted_user_content}
    ]
    
    raw_code_response = generation_agent.get_response(code_gen_prompt)
    save_response(task_id, raw_code_response, "raw_code")
    
    # Clean up the generated code to make sure it's just the implementation
    generated_function_implementation = clean_generated_code(raw_code_response)
    save_response(task_id, generated_function_implementation, "cleaned_code")
    
    # Step 3: Test the code implementation
    code_to_test = task_lean_code.replace("{{code}}", generated_function_implementation).replace("{{proof}}", "sorry")
    
    test_result = execute_lean_code(code_to_test)
    save_response(task_id, test_result, "code_test")
    
    # Step 4: If there's an error, try to fix it
    max_attempts = 1
    attempt = 1
    
    while "Error" in test_result and attempt <= max_attempts:
        # Use verification agent to analyze the error
        verification_template = """
            My Lean 4 code has an error. Here's the original code:

            {code_to_test}

            Here's the error message:

            {test_result}
            
            {examples}

            Please analyze this error step-by-step:
            1. Identify the specific part of the code causing the error
            2. Explain why this is causing an issue
            3. Consider the type system and Lean 4 requirements
            4. Think through potential solutions
            5. Test each solution in your head

            For each step, explicitly share your reasoning process.
            After completing your analysis, provide a fixed version of the function implementation only.
        """
        
        # Add relevant examples if available
        verification_examples_text = ""
        if relevant_examples:
            verification_examples_text = "Here are some relevant examples from the Lean 4 documentation that might help:\n\n" + "\n\n".join(relevant_examples)
        
        # Format the verification content
        formatted_verification_content = verification_template.format(
            code_to_test=code_to_test,
            test_result=test_result,
            examples=verification_examples_text
        )
        
        fix_prompt = [
            {"role": "system", "content": "You are a Lean 4 error diagnostics expert. Your task is to identify and fix errors in Lean code."},
            {"role": "user", "content": formatted_verification_content}
        ]
        
        fix_suggestion = verification_agent.get_response(fix_prompt)
        save_response(task_id, fix_suggestion, f"code_fix_attempt_{attempt}")
        
        # Extract the fixed implementation
        generated_function_implementation = clean_generated_code(fix_suggestion)
        save_response(task_id, generated_function_implementation, f"cleaned_code_fix_{attempt}")
        
        # Test the fixed code
        code_to_test = task_lean_code.replace("{{code}}", generated_function_implementation).replace("{{proof}}", "sorry")
        test_result = execute_lean_code(code_to_test)
        save_response(task_id, test_result, f"code_test_attempt_{attempt}")
        
        attempt += 1
    
    # Step 5: Generate the proof once the code implementation is working
    
    # Prepare proof generation prompt content
    proof_user_content = """
        I need to prove that my Lean 4 implementation satisfies its specification. Here's the full code with the working implementation:

        {code_to_test}
        
        {examples}

        IMPORTANT FORMATTING INSTRUCTIONS:
        1. You MUST generate ONLY the proof code that will directly replace the {{proof}} placeholder in the template.
        2. DO NOT include any surrounding code, explanations, markdown formatting, or comments.
        3. DO NOT include the 'by' keyword - it's already in the template.
        4. DO NOT repeat the 'unfold' line - it's already in the template.
        5. Your response must ONLY contain valid Lean 4 proof tactics.
        6. The placeholder is between the comment lines "-- << PROOF START >>" and "-- << PROOF END >>".

        Example of what I WANT:
        ```
        simp
        cases x
        · simp
        · apply Nat.le_refl
        ```

        Example of what I DON'T want:
        ```lean
        unfold myFunction myFunction_spec
        -- Your proof here
        simp
        ...
        ```

        Now, please provide ONLY the proof content that should replace {{proof}}:
    """
    
    # Add relevant examples if available
    proof_examples_text = ""
    if proof_examples:
        proof_examples_text = "Here are some relevant proof examples that might help:\n\n" + "\n\n".join(proof_examples)
    
    # Format the proof prompt content
    formatted_proof_content = proof_user_content.format(
        code_to_test=code_to_test,
        examples=proof_examples_text
    )
    
    proof_gen_prompt = [
        {"role": "system", "content": "You are a Lean 4 proof expert. Your task is to generate a formal proof for a given implementation. YOU MUST ONLY GENERATE THE EXACT CONTENT THAT SHOULD REPLACE THE {{proof}} PLACEHOLDER - NOTHING MORE, NOTHING LESS."},
        {"role": "user", "content": formatted_proof_content}
    ]
    
    raw_proof = generation_agent.get_response(proof_gen_prompt)
    save_response(task_id, raw_proof, "raw_proof")
    
    # Clean up the generated proof
    generated_proof = clean_generated_code(raw_proof)
    save_response(task_id, generated_proof, "cleaned_proof")
    
    # Step 6: Test the full implementation with the proof
    full_code = task_lean_code.replace("{{code}}", generated_function_implementation).replace("{{proof}}", generated_proof)
    
    full_test_result = execute_lean_code(full_code)
    save_response(task_id, full_test_result, "full_test")
    
    # Step 7: If there's an error with the proof, try to fix it
    proof_attempt = 1
    
    while "Error" in full_test_result and proof_attempt <= max_attempts:
        # First, use verification agent to analyze the error and suggest improvements
        error_analysis_template = """
            I'm trying to prove a theorem in Lean 4, but my proof has an error. Please analyze what's wrong.
            
            Here's the full code including the implementation and proof:

            {full_code}

            Here's the error message:

            {full_test_result}
            
            {examples}
            
            Please analyze this proof error step-by-step:
            1. Identify which tactic is failing and why
            2. Consider the proof state at that point
            3. Check if the proof approach matches the specification
            4. Identify any type mismatches or incorrect assumptions
            5. Consider alternative proof strategies

            For each step, explicitly share your reasoning process.
            Conclude with a detailed analysis of the root cause and suggest a direction for fixing it.
        """
        
        # Add relevant examples if available
        analysis_examples_text = ""
        if proof_examples:
            analysis_examples_text = "Here are some relevant examples from the Lean 4 documentation that might help:\n\n" + "\n\n".join(proof_examples)
        
        # Format the analysis prompt
        formatted_analysis_content = error_analysis_template.format(
            full_code=full_code,
            full_test_result=full_test_result,
            examples=analysis_examples_text
        )
        
        error_analysis_prompt = [
            {"role": "system", "content": "You are a Lean 4 error diagnostics expert. Analyze the error in the proof and explain the issue in detail."},
            {"role": "user", "content": formatted_analysis_content}
        ]
        
        # Get analysis from verification agent
        error_analysis = verification_agent.get_response(error_analysis_prompt)
        save_response(task_id, error_analysis, f"proof_error_analysis_{proof_attempt}")
        
        # Use generation agent to fix the proof, with help from the verification agent's analysis
        proof_fix_template = """
            My Lean 4 proof has an error. Here's the full code including the implementation and proof:

            {full_code}

            Here's the error message:

            {full_test_result}
            
            A Lean 4 expert has analyzed the error and provided this feedback:
            
            {error_analysis}
            
            {examples}

            IMPORTANT FORMATTING INSTRUCTIONS:
            1. You MUST provide ONLY the corrected proof code that will directly replace the {{proof}} placeholder.
            2. DO NOT include any surrounding code, explanations, markdown formatting, or comments.
            3. DO NOT include the 'by' keyword or the 'unfold' line - they're already in the template.
            4. Your response must ONLY contain valid Lean 4 proof tactics.
            5. The {{proof}} placeholder is between the comment lines "-- << PROOF START >>" and "-- << PROOF END >>".

            Please provide ONLY the corrected proof content:
        """
        
        # Format the fix prompt
        formatted_fix_content = proof_fix_template.format(
            full_code=full_code,
            full_test_result=full_test_result,
            error_analysis=error_analysis,
            examples=analysis_examples_text
        )
        
        proof_fix_prompt = [
            {"role": "system", "content": "You are a Lean 4 proof expert. Your task is to fix errors in proofs based on error analysis. YOU MUST ONLY GENERATE THE EXACT CONTENT THAT SHOULD REPLACE THE {{proof}} PLACEHOLDER."},
            {"role": "user", "content": formatted_fix_content}
        ]
        
        proof_fix = generation_agent.get_response(proof_fix_prompt)
        save_response(task_id, proof_fix, f"proof_fix_attempt_{proof_attempt}")
        
        # Extract the fixed proof
        generated_proof = clean_generated_code(proof_fix)
        save_response(task_id, generated_proof, f"cleaned_proof_fix_{proof_attempt}")
        
        # Test the fixed proof
        full_code = task_lean_code.replace("{{code}}", generated_function_implementation).replace("{{proof}}", generated_proof)
        full_test_result = execute_lean_code(full_code)
        save_response(task_id, full_test_result, f"full_test_attempt_{proof_attempt}")
        
        proof_attempt += 1
    
    # Save the final solution
    final_solution = {
        "code": generated_function_implementation,
        "proof": generated_proof
    }
    save_response(task_id, json.dumps(final_solution, indent=2), "final_solution")
    
    return final_solution

def extract_info_from_template(task_lean_code: str) -> Tuple[str, str, List[str]]:
    """
    Extract useful information from the Lean template.
    
    Returns:
        Tuple containing (function_name, specification_definition, parameters)
    """
    # Extract function name
    function_name_match = re.search(r'def\s+(\w+)\s*\(', task_lean_code)
    function_name = function_name_match.group(1) if function_name_match else ""
    
    # Extract specification definition
    spec_def_match = re.search(r'def\s+\w+_spec.*?:=\s*\n\s*--\s*<<\s*SPEC\s*START\s*>>\s*\n(.*?)\n\s*--\s*<<\s*SPEC\s*END', task_lean_code, re.DOTALL)
    spec_def = spec_def_match.group(1).strip() if spec_def_match else ""
    
    # Extract function parameters
    params_match = re.search(r'def\s+\w+\s*\((.*?)\)\s*:', task_lean_code)
    params_str = params_match.group(1) if params_match else ""
    parameters = [p.strip() for p in params_str.split(")") if p.strip()]
    
    return function_name, spec_def, parameters

def clean_generated_code(generated_code: str) -> str:
    """
    Clean up the generated code or proof by removing any markdown code blocks, 
    extra explanation, or formatting that might be included by the LLM.
    """
    # Remove markdown code blocks if present
    code_block_pattern = r'```(?:lean|plaintext)?(.*?)```'
    code_blocks = re.findall(code_block_pattern, generated_code, re.DOTALL)
    
    if code_blocks:
        # Use the largest code block (which is likely the most complete)
        generated_code = max(code_blocks, key=len).strip()
    
    # Remove any lines that seem like comments, explanations, or prefixes
    lines = generated_code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if not line.strip().startswith(('Here', 'The code', 'This is', 'Implementation:', 'Proof:', 'For the', '--')):
            cleaned_lines.append(line)
    
    cleaned_code = '\n'.join(cleaned_lines).strip()
    
    # If the code still has comments or explanations, try to extract just the code
    if "def " in cleaned_code and (function_name_match := re.search(r'def\s+(\w+)', cleaned_code)):
        # If it looks like the LLM included the full function definition, extract just the body
        func_body_match = re.search(r'def\s+\w+.*?:=\s*(.*)', cleaned_code, re.DOTALL)
        if func_body_match:
            cleaned_code = func_body_match.group(1).strip()
    
    return cleaned_code

def get_problem_and_code_from_taskpath(task_path: str) -> Tuple[str, str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and also read the file 
    that contains the task description, which is "description.txt".
    
    After reading the files, it will return a tuple of the problem description and the Lean code template.
    
    Args:
        task_path: Path to the task file
    """
    problem_description = ""
    lean_code_template = ""
    
    with open(os.path.join(task_path, "description.txt"), "r") as f:
        problem_description = f.read()

    with open(os.path.join(task_path, "task.lean"), "r") as f:
        lean_code_template = f.read()

    return problem_description, lean_code_template

def get_unit_tests_from_taskpath(task_path: str) -> List[str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "tests.lean" and return the unit tests.
    """
    with open(os.path.join(task_path, "tests.lean"), "r") as f:
        unit_tests = f.read()
    
    return unit_tests

def get_task_lean_template_from_taskpath(task_path: str) -> str:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and return the Lean code template.
    """
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        task_lean_template = f.read()
    return task_lean_template