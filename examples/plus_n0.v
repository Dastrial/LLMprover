From Stdlib Require Import PeanoNat.

Lemma plus_n0 : forall n : nat, n + 0 = n.
Proof.
induction n as [|n IHn].
- reflexivity.
- cbn. rewrite IHn. reflexivity.
Qed.