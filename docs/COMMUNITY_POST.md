Post this on: https://huggingface.co/princeton-nlp/queen_pawn-8/discussions (New discussion)

Title: Unofficial portable runner (Mac / smaller GPUs) — happy to adjust or take down

Hi! Thanks for releasing QUEEN, the paper and models are great.

I make tutorials on running AI locally, and I've put together an unofficial
open-source wrapper so more people can try QUEEN at home: a small local server,
a drop-in move provider for chess apps, and work in progress on a non-vLLM
backend for Apple Silicon and lower-VRAM NVIDIA cards.

- Repo: <GITHUB LINK>
- It contains only my own code. Weights and your inference code are downloaded
  from your Hugging Face repos at a pinned commit, with links to the paper and repos.
- It's non-commercial, clearly marked as unofficial, and credits you throughout.

A couple of questions:
1. Is this okay with you? If you'd like anything changed or removed, just say so
   and I'll do it right away.
2. Down the line I'd like to share a quantized build for people with smaller GPUs.
   Would you be comfortable with that being hosted on Hugging Face (linked back to
   your model as its base), or would you prefer I don't redistribute weights?

If people end up playing with it, I'd be glad to pass along any human feedback
on the explanations if that's useful for your research.

Thanks again!
— Victor
