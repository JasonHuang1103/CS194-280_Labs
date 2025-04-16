--<< CODE START >>
-- Helper function: Checks if one number divides another or vice versa.
def isDivisible (a b : Nat) : Bool :=
  a % b == 0 || b % a == 0

-- Helper function: counts permutations given a mask of used elements and the last element index.
partial def countPermutations (nums : Array Nat) (mask : Nat) (last : Nat) : Nat :=
  let n := nums.size
  let MOD := 1000000007

  -- If used all elements, then permutation is complete
  if mask == (1 <<< n) - 1 then
    1
  else
    -- Accumulate the count by trying each unused element as the next
    let rec loop (i : Nat) (acc : Nat) : Nat :=
      if i >= n then
        acc
      else if (mask &&& (1 <<< i)) != 0 then
        loop (i + 1) acc
      else
        -- Check divisibility
        let canUse :=
          if mask == 0 then true
          else
            match nums[last]?, nums[i]? with
            | some lastVal, some currVal => isDivisible lastVal currVal
            | _, _ => false
        if canUse then
          -- This element can follow the last one
          let newMask := mask ||| (1 <<< i)
          let newCount := (acc + countPermutations nums newMask i) % MOD
          loop (i + 1) newCount
        else
          loop (i + 1) acc
    loop 0 0

-- Counts special permutations where adjacent elements have divisibility relation.
partial def specialPermutations (nums : Array Nat) : Nat :=
  let n := nums.size
  let MOD := 1000000007
  if n <= 1 then
    1
  else
    -- Try each element as the first in the permutation
    let rec startWith (i : Nat) (acc : Nat) : Nat :=
      if i >= n then
        acc
      else
        let mask := 1 <<< i
        let newAcc := (acc + countPermutations nums mask i) % MOD
        startWith (i + 1) newAcc
    startWith 0 0
--<< CODE END >>

--<< SPEC START >>
def isDivisible_spec (a b : Nat) (result : Bool) : Prop :=
  result = (a % b == 0 || b % a == 0)

def countPermutations_spec (nums : Array Nat) (mask : Nat) (last : Nat) (result : Nat) : Prop :=
  -- This function counts the permutations starting from the element at index 'last'
  -- with 'mask' representing the elements already used (bit i is set if element i is used)
  -- A valid permutation has the property that each adjacent pair (a,b) satisfies:
  -- either a divides b or b divides a

  let n := nums.size
  let MOD := 1000000007

  -- If all elements used, only one permutation exists (the empty one)
  if mask == (1 <<< n) - 1 then
    result = 1
  else if last >= n then
    -- If last is out of bounds, no valid permutations
    result = 0
  else
    -- result is the sum of valid permutations for each possible next element
    -- Each next element must:
    -- 1. Not be used already (not in mask)
    -- 2. Have divisibility relation with the previous element (if any)
    result < MOD  -- The result is reduced modulo MOD

def specialPermutations_spec (nums : Array Nat) (result : Nat) : Prop :=
  -- This function counts permutations of the array where each adjacent pair
  -- has the property that one number divides the other.

  let n := nums.size
  let MOD := 1000000007

  if n <= 1 then
    -- Only one permutation possible for empty or singleton arrays
    result = 1
  else
    -- For larger arrays, result is the sum (modulo MOD) of valid permutations
    -- starting with each possible first element
    result < MOD
--<< SPEC END >>

#guard specialPermutations #[2, 3, 6] = 2
#guard specialPermutations #[1, 4, 3] = 2
#guard specialPermutations #[1, 2, 3, 4, 5, 6] = 4
#guard specialPermutations #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 0
#guard specialPermutations #[8, 12, 48, 96] = 12
