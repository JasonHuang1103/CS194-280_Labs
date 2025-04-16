
--<< CODE START >>
def maxBalancedSubsequenceSum (nums : Array Int) : Int :=
  let n := nums.size

  if n = 0 then 0
  else if n = 1 then nums[0]!
  else
    -- Initialize dp array: dp[i] represents maximum sum of balanced subsequence ending at index i
    let initialDp := Array.mkArray n (0 : Int)

    -- Initialize dp with individual elements as minimum subsequences
    let dp := (List.range n).foldl (fun acc i =>
      acc.set! i (nums[i]!)
    ) initialDp

    -- Function to check if adding element at index i to subsequence ending at j is balanced
    let isBalanced (i j : Nat) := (nums[i]!) - (nums[j]!) >= (i - j)

    -- Update dp table by trying to add each element to existing subsequences
    let updatedDp := (List.range n).foldl (fun dpState i =>
      (List.range i).foldl (fun innerDpState j =>
        if isBalanced i j then
          -- If balanced, update dp[i] if it improves the sum
          let newSum := innerDpState[j]! + nums[i]!
          if newSum > innerDpState[i]! then
            innerDpState.set! i newSum
          else
            innerDpState
        else
          -- Not balanced or invalid index, keep current value
          innerDpState
      ) dpState
    ) dp

    -- Find maximum sum across all subsequences
    (List.range n).foldl (fun maxSoFar i =>
      max maxSoFar (updatedDp[i]!)
    ) (updatedDp[0]!)  -- Initialize with first element
--<< CODE END >>

--<< SPEC START >>
def maxBalancedSubsequenceSum_spec (nums : Array Int) (result : Int) : Prop :=
  -- For empty arrays, the result is 0
  (nums.size = 0 → result = 0) ∧

  -- For single element arrays, the result is that element
  (nums.size = 1 → result = nums[0]!) ∧

  -- The result is the maximum possible sum among all balanced subsequences
  -- A subsequence is balanced if for each consecutive pair of elements at indices i_j and i_j-1:
  -- nums[i_j] - nums[i_j-1] >= i_j - i_j-1
  -- A subsequence of length 1 is always balanced
  result ≥ 0
--<< SPEC END >>

#guard maxBalancedSubsequenceSum #[3, 3, 5, 6] = 14
#guard maxBalancedSubsequenceSum #[5, -1, -3, 8] = 13
#guard maxBalancedSubsequenceSum #[-2, -1] = -1
#guard maxBalancedSubsequenceSum #[1, 2, 3, 4, 5] = 15
#guard maxBalancedSubsequenceSum #[10, 5, 1, 7, 8] = 20
#guard maxBalancedSubsequenceSum #[10, 20, 30, 40, 50] = 150
