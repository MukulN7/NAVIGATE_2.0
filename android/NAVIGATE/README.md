# NAVIGATE Android Application Foundation

Native Android application foundation for SIH26168 NAVIGATE 2.0 (GNSS-Denied Intelligent Navigation).

---

## Project Specifications

- **Language**: Kotlin 1.9.22
- **Minimum SDK**: API 26 (Android 8.0 Oreo)
- **Target SDK**: API 34 (Android 14)
- **Build System**: Gradle 8.2 with Kotlin DSL (`build.gradle.kts`)
- **UI Architecture**: Single Activity (`MainActivity.kt`) with ViewBinding & Material Components

---

## How to Open in Android Studio

1. Download and install **Android Studio** (Hedgehog 2023.1.1 or newer recommended).
2. Open Android Studio and click **Open** (or select `File > Open...`).
3. Select the project folder:
   `NAVIGATE_2.0/android/NAVIGATE`
4. Click **OK**.
5. Android Studio will automatically configure the JDK, download the required Android SDK Build Tools (API 34), and sync the Gradle project.

---

## How to Build the Application

### Option A: Using Android Studio (Recommended)
1. Open the project in Android Studio.
2. Click **Build > Make Project** (or press `Ctrl + F9` / `Cmd + F9`).
3. To generate a debug APK: Select **Build > Build Bundle(s) / APK(s) > Build APK(s)**.

### Option B: Using Gradle Wrapper (Command Line)
Navigate to the `android/NAVIGATE` directory:
```bash
./gradlew assembleDebug
```
*(On Windows PowerShell: `.\gradlew.bat assembleDebug`)*

The built debug APK will be generated at:
`app/build/outputs/apk/debug/app-debug.apk`

---

## How to Run on a Physical Android Phone

1. **Enable Developer Options & USB Debugging on your Phone**:
   - Open **Settings > About Phone**.
   - Tap **Build Number** 7 times until you see the toast *"You are now a developer!"*.
   - Navigate to **Settings > System > Developer Options**.
   - Toggle on **USB Debugging**.

2. **Connect Device**:
   - Connect your Android smartphone to your computer using a USB cable.
   - When prompted on your phone screen, tap **"Allow USB debugging"**.

3. **Run from Android Studio**:
   - Select your physical phone from the top toolbar device selector dropdown.
   - Click the green **Run** button (or press `Shift + F10`).

4. **Run via ADB (Command Line)**:
   ```bash
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   adb shell am start -n com.sih26168.navigate/.MainActivity
   ```
