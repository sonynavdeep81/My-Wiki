# Slide Deck Format (load when the user asks for a slide deck)

Single `.md` file. Never .pptx / build scripts / image exports.

**Slide 1 = Cover, exactly:**

```
## Slide 1 — Cover

# [Presentation Title]

*[Subtitle]*

**Dr. Navdeep Singh**
ASSOCIATE PROFESSOR
*Computer Science & Engineering · Punjabi University, Patiala*
```

**Other slides:** `## Slide N — Title`; plain bullets, easy to understand; **no tables ever**; renderable by Gamma/Gemini/Claude design. Follow `learning-path.md` stage order; silently skip stages under `## Gaps in This Path`.
