/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // спокойная студийная тема:长 работа с таймлайном на светлом фоне утомляет
        ink: { 900: "#0d0f13", 800: "#14171d", 700: "#1c2029", 600: "#252a35", 500: "#333a48" },
        line: "#2c3240",
        muted: "#8b93a5",
        accent: "#4c8dff",
        warn: "#ffb020",
        danger: "#f2585b",
        ok: "#33b679",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
