import Std.Data.HashMap

--<< CODE START >>
partial def canPartition (numStr : String) (target : Nat) : Bool :=
  let rec check (start : Nat) (remaining : Nat) : Bool :=
    if start == numStr.length then
      remaining == 0
    else
      let rec tryPartition (i : Nat) : Bool :=
        if i > numStr.length then
          false
        else
          let subStr := numStr.extract (String.Pos.mk start) (String.Pos.mk i)
          match String.toNat? subStr with
          | none => tryPartition (i + 1)
          | some val =>
              if val > remaining then
                tryPartition (i + 1)
              else if check i (remaining - val) then
                true
              else
                tryPartition (i + 1)
      tryPartition (start + 1)
  check 0 target

partial def punishmentNumber (n : Nat) : Nat :=
  let rec sumPunishments (i : Nat) (acc : Nat) : Nat :=
    if i > n then
      acc
    else
      let square := i * i
      let squareStr := toString square
      if canPartition squareStr i then
        sumPunishments (i + 1) (acc + square)
      else
        sumPunishments (i + 1) acc
  sumPunishments 1 0
--<< CODE END >>

--<< SPEC START >>
def canPartition_spec (numStr : String) (target : Nat) (result : Bool) : Prop :=
  -- This function checks if the digits of numStr can be partitioned into a sequence of integers
  -- that sum up to target

  -- result is true if and only if there exists a partition of the digits in numStr
  -- such that the sum of the resulting integers equals target
  result =
    ∃ partition : List Nat,
      -- The partition represents a way to split numStr
      -- The concatenation of all digits in the partition equals numStr
      -- And the sum of all elements in the partition equals target
      (partition.foldl (· + ·) 0 = target) ∧
      -- partition is a valid way to split numStr into integers
      let digitsStr := partition.map toString
      String.join digitsStr = numStr

def punishmentNumber_spec (n : Nat) (result : Nat) : Prop :=
  -- This function finds the sum of all "punishment numbers" up to n
  -- A punishment number is a number i where i² can be partitioned into a sum that equals i

  -- For all valid inputs, result equals the sum of squares of all punishment numbers
  let nums := List.range (n+1)
  let numsFrom1 := List.drop 1 nums
  let punishmentNumbers := List.filter (fun i =>
    let square := i * i
    let squareStr := toString square
    canPartition squareStr i) numsFrom1
  let sumOfSquares := List.foldl (fun acc i => acc + i*i) 0 punishmentNumbers

  result = sumOfSquares
--<< SPEC END >>

#guard punishmentNumber 10 = 182
#guard punishmentNumber 37 = 1478
#guard punishmentNumber 99 = 31334
#guard punishmentNumber 489 = 772866
#guard punishmentNumber 1000 = 10804657
