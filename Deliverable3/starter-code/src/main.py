import os
import argparse
from src.agents import Reasoning_Agent, LLM_Agent
from src.lean_runner import execute_lean_code
from src.embedding_db import VectorDB, OpenAIEmbeddingModel
from typing import Dict, List, Tuple, TypeAlias

# type LeanCode = Dict[str, str]
LeanCode: TypeAlias = Dict[str, str]

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
    # generated_implementation = "sorry"
    # generated_proof = "sorry"

    # # TODO Implement your coding workflow here. The unit tests will call this function as the main workflow.
    # # Feel free to chain multiple agents together, use the RAG database (.pkl) file, corrective feedback, etc.
    # # Please use the agents provided in the src/agents.py module, which include GPT-4o and the O3-mini models.
    # ...

    # # Example return for task_id_0
    # generated_implementation = "x"
    # generated_proof = "rfl"

    # Initialize agents
    planning_agent = Reasoning_Agent(model="o3-mini")  # For task decomposition and strategy
    generation_agent = LLM_Agent(model="gpt-4o")  # For code and proof generation
    verification_agent = Reasoning_Agent(model="o3-mini")  # For error analysis and feedback

    # Initialize RAG database
    embedding_model = OpenAIEmbeddingModel()
    vector_db = VectorDB(directory="documents", 
                        vector_file="database.npy", 
                        embedding_model=embedding_model)

    # Step 1: Planning Phase
    planning_prompt = f"""
    Given the following problem description and Lean template, create a detailed plan for implementing the solution:
    
    Problem Description:
    {problem_description}
    
    Lean Template:
    {task_lean_code}
    
    Your plan should include:
    1. Key requirements and constraints
    2. High-level approach for implementation
    3. Potential challenges and how to address them
    4. Required Lean tactics and theorems
    """
    
    planning_messages = [
        {"role": "system", "content": "You are an expert Lean 4 programmer and theorem prover."},
        {"role": "user", "content": planning_prompt}
    ]
    plan = planning_agent.get_response(planning_messages)

    # Step 2: Retrieve relevant examples using RAG
    relevant_examples = vector_db.get_top_k("database.npy", embedding_model, problem_description, k=3)

    # Step 3: Generation Phase
    generation_prompt = f"""
    Based on the following plan and relevant examples, implement the solution in Lean 4:
    
    Plan:
    {plan}
    
    Relevant Examples:
    {relevant_examples}
    
    Generate both the implementation code and the proof. The code should replace {{code}} and the proof should replace {{proof}}.
    """
    
    generation_messages = [
        {"role": "system", "content": "You are an expert Lean 4 programmer and theorem prover."},
        {"role": "user", "content": generation_prompt}
    ]
    solution = generation_agent.get_response(generation_messages)

    # Extract code and proof from the solution
    generated_implementation = solution.split("{{code}}")[1].split("{{proof}}")[0].strip()
    generated_proof = solution.split("{{proof}}")[1].strip()

    # Step 4: Verification and Feedback Loop
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        # Test the implementation
        test_code = task_lean_code.replace("{{code}}", generated_implementation).replace("{{proof}}", "sorry")
        result = execute_lean_code(test_code)
        
        if "Lean code executed successfully" in result:
            # Test the proof
            full_code = task_lean_code.replace("{{code}}", generated_implementation).replace("{{proof}}", generated_proof)
            result = execute_lean_code(full_code)
            
            if "Lean code executed successfully" in result:
                break
            else:
                # Proof failed, get feedback
                feedback_prompt = f"""
                The proof failed with the following error:
                {result}
                
                Current implementation:
                {generated_implementation}
                
                Current proof:
                {generated_proof}
                
                Please analyze the error and suggest corrections.
                """
                
                feedback_messages = [
                    {"role": "system", "content": "You are an expert Lean 4 theorem prover."},
                    {"role": "user", "content": feedback_prompt}
                ]
                feedback = verification_agent.get_response(feedback_messages)
                
                # Update proof based on feedback
                proof_prompt = f"""
                Based on the following feedback, correct the proof:
                
                Feedback:
                {feedback}
                
                Current implementation:
                {generated_implementation}
                
                Current proof:
                {generated_proof}
                """
                
                proof_messages = [
                    {"role": "system", "content": "You are an expert Lean 4 theorem prover."},
                    {"role": "user", "content": proof_prompt}
                ]
                corrected_proof = generation_agent.get_response(proof_messages)
                generated_proof = corrected_proof.strip()
        else:
            # Implementation failed, get feedback
            feedback_prompt = f"""
            The implementation failed with the following error:
            {result}
            
            Current implementation:
            {generated_implementation}
            
            Please analyze the error and suggest corrections.
            """
            
            feedback_messages = [
                {"role": "system", "content": "You are an expert Lean 4 programmer."},
                {"role": "user", "content": feedback_prompt}
            ]
            feedback = verification_agent.get_response(feedback_messages)
            
            # Update implementation based on feedback
            implementation_prompt = f"""
            Based on the following feedback, correct the implementation:
            
            Feedback:
            {feedback}
            
            Current implementation:
            {generated_implementation}
            """
            
            implementation_messages = [
                {"role": "system", "content": "You are an expert Lean 4 programmer."},
                {"role": "user", "content": implementation_prompt}
            ]
            corrected_implementation = generation_agent.get_response(implementation_messages)
            generated_implementation = corrected_implementation.strip()
        
        attempt += 1

    return {
        "code": generated_implementation,
        "proof": generated_proof
    }

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