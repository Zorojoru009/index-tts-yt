# IndexTTS Script Format Guide

This guide explains how to properly format scripts for optimal text-to-speech generation with IndexTTS.

## Overview

Scripts should be plain text files (`.txt`) with proper formatting to ensure:
- Natural speech flow and pacing
- Correct pronunciation
- Appropriate emotional delivery
- No synthesis errors

---

## Basic Rules

### 1. **Plain Text Only**
- Use `.txt` format (not `.docx`, `.pdf`, or `.md`)
- No special formatting symbols (bold, italics, underlines)
- No HTML tags or markdown syntax
- Clean ASCII characters only

### 2. **Sentence Structure**
- Use complete sentences with proper punctuation
- Each thought should end with a period, question mark, or exclamation mark
- Avoid sentence fragments (unless intentional for effect)

**❌ Bad:**
```
rotting. lazy. stuck.
```

**✅ Good:**
```
You are rotting. You are lazy. You are stuck.
```

### 3. **Quotation Marks**
- Use straight quotation marks: `"` and `"`
- Avoid fancy quotes: `"` and `"`
- Ensure matching pairs

**❌ Bad:**
```
He said, "This is wrong..."
```

**✅ Good:**
```
He said, "This is wrong."
```

### 4. **Contractions**
- Use standard English contractions: `don't`, `can't`, `won't`, `you're`, `it's`
- TTS engines read contractions naturally and correctly
- Avoid writing out contractions as separate words

**❌ Bad:**
```
Do not worry. You will be fine.
```

**✅ Good:**
```
Don't worry. You'll be fine.
```

### 5. **Numbers**
- Write out numbers in words for natural speech
- Use digits only for: phone numbers, addresses, years (context-dependent)

**❌ Bad:**
```
In 10 years, you'll have 5 options.
```

**✅ Good:**
```
In ten years, you'll have five options.
```

### 6. **Emphasis**
- **DO NOT use ALL CAPS** - TTS reads caps same as lowercase, but creates visual noise
- **DO NOT use repeating characters** - `REALLY!!!` or `weird...` (overuse)
- Let natural word choice and sentence structure convey emphasis

**❌ Bad:**
```
YOU ARE NOT LOST!!! This is YOUR CHOICE!!!
```

**✅ Good:**
```
You are not lost. This is your choice.
```

### 7. **Punctuation for Pacing**
- Use periods for normal pacing
- Use ellipsis `...` for dramatic pauses (use sparingly)
- Use commas to separate clauses
- Use semicolons to connect related thoughts

**❌ Bad:**
```
You are rotting rotting rotting every single day
```

**✅ Good:**
```
You are rotting. Every single day.
```

### 8. **Abbreviations and Acronyms**
- Spell out abbreviations for clarity: `doctor` instead of `Dr.`, `and` instead of `&`
- For acronyms, write out each letter separately if important
- Common words like `okay`, `hello` should be written out

**❌ Bad:**
```
The CEO & CFO met w/ the Dr. at 3pm EST.
```

**✅ Good:**
```
The chief executive officer and chief financial officer met with the doctor at three p-m eastern standard time.
```

### 9. **Hyphens and Dashes**
- Use hyphens for compound words: `twenty-first`, `self-aware`
- Use dashes (em-dashes) sparingly for breaks in thought
- Don't overuse: `word - word - word`

**❌ Bad:**
```
This - is - important
```

**✅ Good:**
```
This is important. It matters.
```

---

## Paragraph Structure

### Spacing and Breaks
- Use **blank lines between paragraphs** for natural pacing
- Each paragraph should be a complete thought or topic
- Short sentences create impact; long paragraphs create flow

**❌ Bad (no breaks):**
```
You are rotting. You think your depression comes from your past. You think you are this way because of trauma. You are lying to yourself. You tell yourself stories.
```

**✅ Good (natural breaks):**
```
You are rotting.

You think your depression comes from your past. You think you are this way because of trauma.

You are lying to yourself.

You tell yourself stories.
```

### Dialogue and Quotes
- Quote marks should be straight: `"` not `"`
- Short quotes work better than long ones in TTS
- Attribute quotes clearly

**✅ Good:**
```
Adler said, "No experience is a cause of success or failure."

He taught that we create our reality through our choices.
```

---

## Pronunciation Tips

### Names and Proper Nouns
- Write out pronunciation hints in parentheses if needed
- For foreign names, use common English approximations

**Example:**
```
Alfred Adler (ALL-fred AHD-ler) discovered teleology.
```

### Technical Terms
- Define technical terms on first use
- Use simple synonyms when possible

**❌ Bad:**
```
Teleological action drives behavior.
```

**✅ Good:**
```
Teleology is the idea that we are driven by future goals, not past causes.
Teleological action means moving towards a specific goal.
```

### Difficult Words
- Break complex words into syllables if needed
- Use context to help with pronunciation

**Example:**
```
Individual psychology is the study of how unique people create meaning.
```

---

## Emotional Delivery

### Punctuation for Emotion
- `!` for excitement, urgency, anger
- `.` for calm, serious, measured
- `?` for curiosity, questioning
- `...` for dramatic pauses (use sparingly)

**Example - Building intensity:**
```
You are rotting.

You are wasting time.

You are dying.

How much longer will you wait?
```

### Sentence Length for Pacing
- Short sentences = urgency, impact
- Long sentences = flowing, explanatory
- Mix both for dynamic delivery

**✅ Good mix:**
```
You have a choice. This moment right now, you can decide who you want to become.

But it requires sacrifice. It requires discipline. It requires you to kill the part of you that wants to be saved.
```

---

## Emotion Control Integration

Use IndexTTS emotion control with your script:

### Recommended Emotion Modes by Content Type

| Content Type | Mode | Emotion | Strength |
|---|---|---|---|
| Motivational | Vector | Angry: 0.8, Determined: emphasis | 0.7 |
| Story/Narrative | Audio Ref | Natural emotion from reference | 0.6 |
| Calm/Educational | Speaker | Same as speaker | 0.5 |
| Dark/Intense | Vector | Sad: 0.6, Angry: 0.4 | 0.7 |

**Example Script + Emotion:**
The Adler anti-vision script works best with:
- **Emotion Mode**: Vector control
- **Happy**: 0.0 (no happiness)
- **Angry**: 0.85 (intense, driving)
- **Sad**: 0.02 (subtle undertone)
- **Strength**: 0.8

---

## File Organization

### Folder Structure
```
scripts/
├── motivational/
│   ├── adler_antivision.txt
│   ├── discipline.txt
│   └── mindset.txt
├── educational/
│   ├── psychology_101.txt
│   └── history_lesson.txt
├── narrative/
│   └── short_story.txt
└── README.md
```

### Naming Convention
- Use lowercase with underscores: `adler_antivision.txt`
- Include topic/theme: `discipline_motivation.txt`
- Add date if versioning: `antivision_v2_2024.txt`

---

## Common Mistakes to Avoid

| Mistake | Problem | Fix |
|---|---|---|
| ALL CAPS TEXT | Hard to read, no benefit | Use normal capitalization |
| Weird...punctuation... | Awkward pauses | Use standard punctuation |
| `&` symbols | Wrong pronunciation | Use word "and" |
| No paragraph breaks | TTS sounds robotic | Add blank lines |
| Numbers as digits | Mispronunciation | Write out words |
| Fancy quotes "" | Encoding issues | Use straight " |
| Run-on sentences | Loss of pacing | Use periods and short sentences |
| Too many ellipsis | Annoying pauses | Use 1-2 max per script |
| No punctuation at all | Unclear flow | Add proper periods/commas |
| Abbreviations | Confusing pronunciation | Write out full words |

---

## Quick Checklist

Before uploading your script:

- [ ] Saved as `.txt` file (plain text)
- [ ] No special formatting (bold, italic, underline)
- [ ] All quotation marks are straight: `"`
- [ ] Numbers written as words (ten, twenty, etc.)
- [ ] Contractions used naturally (don't, can't, you're)
- [ ] No ALL CAPS sections
- [ ] Paragraph breaks between thoughts
- [ ] Proper sentence structure (subject + verb + period)
- [ ] No fancy symbols (`&`, `@`, `#`, `*`, etc.)
- [ ] Dialogue clearly attributed
- [ ] Consistent tone throughout
- [ ] Proofread for spelling errors

---

## Example: Before and After

### ❌ BEFORE (Poorly Formatted)
```
YOU ARE not "lost." YOU ARE not "healing." And YOU ARE certainly not "waiting for the right moment."

YOU ARE rotting.

You think your depression, your laziness, your lack of drive comes from your past...

You think YOU ARE this way, because of what happened to you...

Because of your trauma...

Because of your parents.

YOU ARE LYING to yourself.

You tell yourself stories...

"I am an introvert." "I am not good with people." "I have a slow metabolism."
These are not fACTs...

These are cages...

And you built them.
```

### ✅ AFTER (Properly Formatted)
```
You are not lost. You are not healing. And you are certainly not waiting for the right moment.

You are rotting.

You think your depression, your laziness, your lack of drive comes from your past. You think you are this way because of what happened to you. Because of your trauma. Because of your parents.

You are lying to yourself.

You tell yourself stories. I am an introvert. I am not good with people. I have a slow metabolism. These are not facts. These are cages. And you built them.
```

---

## Tips for Best Results

### 1. **Read Aloud**
Read your script out loud before uploading. If it sounds awkward spoken, it will sound awkward in TTS.

### 2. **Vary Sentence Length**
Mix short impactful sentences with longer explanatory ones for natural rhythm.

### 3. **Use Paragraph Breaks Strategically**
- After a major point
- Before a question
- When changing topics
- To emphasize contrast

### 4. **Simple Words Over Complex**
- Use "help" instead of "facilitate"
- Use "buy" instead of "procure"
- Use "sad" instead of "melancholic"

### 5. **Consider Emotion Control**
- Script content should match the emotion mode
- Angry scripts: short, punchy sentences
- Calm scripts: longer, flowing sentences
- Sad scripts: reflective, questioning tone

---

## Testing Your Script

1. **Upload to IndexTTS**
2. **Generate with different emotions** to see how delivery changes
3. **Listen carefully** for:
   - Natural pacing
   - Clear pronunciation
   - Appropriate emphasis
   - Smooth transitions between paragraphs
4. **Make adjustments** if needed (add punctuation, break up sentences, etc.)
5. **Download and save** the final audio

---

## Resources

- [IndexTTS Multi-GPU Documentation](./multi_gpu.md)
- [Emotion Control Guide](./multi_gpu.md#emotion-control-implementation)
- [WebUI Usage Guide](./README.md)

---

**Happy scripting! 🎙️**
