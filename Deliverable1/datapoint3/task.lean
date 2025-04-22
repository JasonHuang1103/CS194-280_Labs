--<< CODE START >>
partial def hasEnoughDistinct (nums : Array Nat) (start : Nat) (len : Nat) (m : Nat) : Bool :=
  let slice := nums.extract start (start + len)

  -- Count distinct elements using a simple approach
  let rec countDistinct (i : Nat) (seen : List Nat) : Nat :=
    if i >= slice.size then
      seen.length
    else
      let curr := slice[i]!
      if seen.contains curr then
        countDistinct (i + 1) seen
      else
        countDistinct (i + 1) (curr :: seen)

  countDistinct 0 [] >= m

partial def sliceSum (nums : Array Nat) (start : Nat) (len : Nat) : Nat :=
  let slice := nums.extract start (start + len)
  let rec sumElements (i : Nat) (acc : Nat) : Nat :=
    if i >= slice.size then
      acc
    else
      sumElements (i + 1) (acc + slice[i]!)
  sumElements 0 0

partial def maxSum (nums : Array Nat) (m : Nat) (k : Nat) : Nat :=
  if nums.size < k then
    0
  else
    -- Check each possible subarray of length k
    let rec findMaxSum (start : Nat) (currMax : Nat) : Nat :=
      if start + k > nums.size then
        currMax
      else
        -- Check if current window has enough distinct elements
        let isValid := hasEnoughDistinct nums start k m
        let currSum := sliceSum nums start k
        -- Update max sum if valid
        let newMax := if isValid then max currMax currSum else currMax
        -- Move to next window
        findMaxSum (start + 1) newMax
    findMaxSum 0 0
--<< CODE END >>

-- << SPEC START >>
def hasEnoughDistinct_spec (nums : Array Nat) (start : Nat) (len : Nat) (m : Nat) (result : Bool) : Prop :=
  result = if start + len ≤ nums.size then
    -- The function checks if there are at least m distinct elements
    let slice := nums.extract start (start + len)
    let distinctElements := List.foldl
      (fun acc x => if acc.contains x then acc else x :: acc)
      []
      slice.toList
    distinctElements.length ≥ m
  else
    false

def sliceSum_spec (nums : Array Nat) (start : Nat) (len : Nat) (result : Nat) : Prop :=
  result = if start + len ≤ nums.size then
    -- The function sums all elements in the subarray
    let slice := nums.extract start (start + len)
    List.foldl (fun acc x => acc + x) 0 slice.toList
  else
    0

def maxSum_spec (nums : Array Nat) (m : Nat) (k : Nat) (result : Nat) : Prop :=
  if nums.size < k then
    result = 0
  else
    -- Result is maximum sum of any valid subarray
    (∃ start : Nat,
      start + k ≤ nums.size ∧
      hasEnoughDistinct nums start k m = true ∧
      sliceSum nums start k = result) ∧
    -- No valid subarray has a larger sum
    (∀ otherStart : Nat,
      otherStart + k ≤ nums.size →
      hasEnoughDistinct nums otherStart k m = true →
      sliceSum nums otherStart k ≤ result)
-- << SPEC END >>

#guard maxSum #[2, 6, 7, 3, 1, 7] 3 4 == 18
#guard maxSum #[5, 9, 9, 2, 4, 5, 4] 1 3 == 23
#guard maxSum #[1, 2, 1, 2, 1, 2, 1] 3 3 == 0
#guard maxSum #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 3 4 == 34
#guard maxSum #[1, 1, 2, 3, 5, 8, 13, 21] 5 5 == 50
