def minOfThree (a : Int) (b : Int) (c : Int) : Int :=
  -- << CODE START >>
  if a <= b && a <= c then a
  else if b <= a && b <= c then b
  else c
  -- << CODE END >>

def minOfThree_spec_isMin (a : Int) (b : Int) (c : Int) (result : Int) : Prop :=
  -- << SPEC START >>
  (result <= a ∧ result <= b ∧ result <= c) ∧ (result = a ∨ result = b ∨ result = c)
  -- << SPEC END >>

-- You can use the following to do unit tests, you don't need to submit the following code

#guard minOfThree 3 2 1 = 1
#guard minOfThree 5 5 5 = 5
#guard minOfThree 10 20 15 = 10
#guard minOfThree (-1) 2 3 = (-1)
#guard minOfThree 2 (-3) 4 = (-3)
#guard minOfThree 2 3 (-5) = (-5)
#guard minOfThree 0 0 1 = 0
#guard minOfThree 0 (-1) 1 = (-1)
#guard minOfThree (-5) (-2) (-3) = (-5)
