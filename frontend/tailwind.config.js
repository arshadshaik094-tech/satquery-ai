/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#0B1220",
          900: "#0F1B33",
          800: "#152343",
          700: "#1C2C4F",
        },
        brand: {
          50: "#EEF3FF",
          100: "#DCE7FF",
          400: "#5B8DFF",
          500: "#2F6BFF",
          600: "#1E52E6",
          700: "#173FB3",
        },
        accent: {
          teal: "#14C3A6",
          amber: "#F5A623",
          red: "#EF4444",
          violet: "#7C6CF0",
        },
        surface: "#F6F8FC",
        card: "#FFFFFF",
        line: "#E6EAF2",
        ink: {
          900: "#0E1626",
          700: "#3A445A",
          500: "#6B7488",
          400: "#8C93A4",
        },
      },
      fontFamily: {
        sans: ["'Plus Jakarta Sans'", "Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(14,22,38,0.04), 0 8px 24px -12px rgba(14,22,38,0.10)",
        pop: "0 12px 32px -8px rgba(47,107,255,0.28)",
      },
      borderRadius: {
        xl2: "1rem",
      },
    },
  },
  plugins: [],
};
