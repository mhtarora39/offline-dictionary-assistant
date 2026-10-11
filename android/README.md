# 📖 Offline AI Dictionary (Android)

A 100% offline, privacy-first dictionary app for Android. Powered by:
- **On-Device NLP (ONNX Runtime Mobile)**: Extracts headwords from natural language queries (e.g. *"what does kick the bucket mean"* $\to$ *"kick the bucket"*) in **< 4 ms**.
- **0.0% Invented Terms**: Uses character offset mappings from the WordPiece tokenizer to slice terms directly from the raw user string.
- **Native Offline Voice-to-Text**: Utilizes Android's on-device `SpeechRecognizer` (`EXTRA_PREFER_OFFLINE = true`) with **0 MB added app size**.
- **Embedded WordNet 3.0 SQLite**: Bundled database with fuzzy typo correction (Levenshtein distance $\le 2$).
- **Modern Jetpack Compose UI**: Fast, responsive dark-themed interface with instant latency feedback.

---

## 📁 Project Structure & Assets

```
android/
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml          # Audio permission & speech service query
│       ├── assets/                      # Bundled offline assets
│       │   ├── term_extractor_int8.onnx # Quantized INT8 ONNX NLP model (22.9 MB)
│       │   ├── vocab.txt                # WordPiece vocabulary (30,522 tokens, 262 KB)
│       │   └── dictionary.db            # WordNet 3.0 SQLite database (45.2 MB)
│       └── java/com/offline/dictionary/
│           ├── MainActivity.kt          # Dynamic mic permission & Compose entry
│           ├── nlp/
│           │   ├── WordPieceTokenizer.kt# On-device WordPiece tokenizer with offsets
│           │   └── TermExtractor.kt     # ONNX Runtime session & subtoken grouping
│           ├── db/
│           │   └── DictionaryDbHelper.kt# Asset copy & SQLite lookup with fuzzy match
│           ├── voice/
│           │   └── OfflineVoiceRecognizer.kt # Native on-device speech-to-text
│           └── ui/
│               ├── DictionaryViewModel.kt # State management & async latency timing
│               └── DictionaryScreen.kt  # Jetpack Compose UI
```

---

## 🛠 Prerequisites

1. **Android Studio**: Android Studio Jellyfish (2023.3+) or Ladybug (2024.1+).
2. **Java / JDK**: JDK 17 (automatically provided by Android Studio under `Android Studio/jbr`).
3. **Android SDK**:
   - Compile SDK: `34` (Android 14)
   - Minimum SDK: `26` (Android 8.0 Oreo)

---

## 🚀 Building & Running in Android Studio

### 1. Open the Project
1. Launch Android Studio.
2. Select **File $\to$ Open...** (or click **Open** on the Welcome screen).
3. Select this `android` folder:
   ```
   C:\Users\mhtar\Desktop\Projects\dictionary\android
   ```
4. Click **OK** (and click **Trust Project** if prompted).
5. Allow Android Studio to complete the initial Gradle Sync.

---

### 2. Create an Emulator (Virtual Device)
1. Open the **Device Manager**:
   - Click the phone icon in the top-right toolbar, or go to **Tools $\to$ Device Manager**.
2. Click **Create Device** (or **+**).
3. Choose **Pixel 8** (or any phone) $\to$ click **Next**.
4. **Important (ABI Choice):**
   - On Windows PCs (Intel or AMD CPU): Select **`x86_64`** (API 33 or 34).
   - *(Why? `x86_64` runs natively with hardware acceleration on Windows. Do not select `arm64-v8a` for a Windows emulator as software translation will lag).*
   - If needed, click **Download** next to the system image $\to$ click **Next** $\to$ **Finish**.

---

### 3. Enable PC Microphone in the Emulator (For Voice Search)
To test voice recognition with your real computer microphone:
1. Start your emulator by clicking the **Play (▶)** button in Device Manager.
2. On the floating emulator toolbar on the side, click the **Three Dots (⋯)** (*Extended Controls*).
3. In the left menu, select **Microphone**.
4. Check the box: **"Virtual microphone uses host audio input"**.
5. Close the Extended Controls window.

---

### 4. Run the App
1. In the top toolbar, select `app` in the configuration dropdown and your emulator as the target.
2. Click the green **Run (▶)** button (or press `Shift + F10`).
3. The app will build, install, and launch automatically.

---

## 🧪 Testing the Features

### 1. Natural Language Typing
- Type complex sentences into the search bar:
  - *"what does kick the bucket mean"* $\to$ Extracts *"kick the bucket"* and shows definitions.
  - *"tell me what is serendipity mean"* $\to$ Extracts *"serendipity"*.
- Try misspellings:
  - *"what is nieghborhood"* $\to$ Autocorrects and returns *"neighborhood"*.
- Notice the latency badge below the search bar: **⚡ 2-4ms NLP • 5ms total**.

### 2. Native Offline Voice Search
1. Tap the **Microphone** icon in the search bar.
2. Accept the permission prompt: *"While using the app"*.
3. Speak clearly into your mic: *"define photosynthesis please"*.
4. The transcript will stream live, and the definition will appear immediately.

### 3. Verify 100% Offline Behavior
1. Swipe down the Android notification shade in the emulator.
2. Turn on **Airplane Mode**.
3. Perform both text searches and voice searches.
4. The app operates completely with zero network access.

---

## 💻 Command Line Build (Optional)

If you prefer building directly from the command line:

```powershell
# Build debug APK
./gradlew assembleDebug

# Install on a connected emulator or phone via ADB
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch the app
adb shell am start -n com.offline.dictionary/.MainActivity
```
