def gcd (a b : Int) : Int :=
  Int.gcd a b

def canTraverseAllPairs (nums : List Int) : Bool :=
  -- << CODE START >>
  let n := nums.length

  -- Base cases
  if n == 1 then
    true
  else if nums.any (fun x => x == 1) then
    false
  else
    -- Simple implementation: check if all pairs have path between them
    let hasPath (i j : Nat) : Bool :=
      -- Direct connection
      if gcd nums[i]! nums[j]! > 1 then true
      -- Check if there's an intermediate node
      else List.any (List.range n) fun k =>
        k != i && k != j && gcd nums[i]! nums[k]! > 1 && gcd nums[k]! nums[j]! > 1

    -- Check all pairs
    List.all (List.range n) fun i =>
      List.all (List.range n) fun j =>
        i == j || hasPath i j
  -- << CODE END >>

def canTraverseAllPairs_spec (nums : List Int) (result : Bool) : Prop :=
  -- << SPEC START >>
  if result then
    ∀ i j, 0 ≤ i ∧ i < nums.length ∧ i < j ∧ j < nums.length →
      ∃ (path : List Nat), path.length ≥ 2 ∧ path.head! = i ∧ path.getLast! = j ∧
      ∀ k, k + 1 < path.length → gcd nums[path[k]]! nums[path[k+1]]! > 1
  else
    ∃ i j, 0 ≤ i ∧ i < nums.length ∧ i < j ∧ j < nums.length ∧
      ¬∃ (path : List Nat), path.length ≥ 2 ∧ path.head! = i ∧ path.getLast! = j ∧
      ∀ k, k + 1 < path.length → gcd nums[path[k]]! nums[path[k+1]]! > 1
  -- << SPEC END >>

-- Unit tests
#guard canTraverseAllPairs [2, 3, 6] = true
#guard canTraverseAllPairs [3, 9, 5] = false
#guard canTraverseAllPairs [4, 3, 12, 8] = true
