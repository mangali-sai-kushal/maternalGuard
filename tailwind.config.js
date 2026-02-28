/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        crimson: {
          50:  '#fff1f2',
          100: '#ffe0e3',
          200: '#ffc5cb',
          300: '#fe98a2',
          400: '#fc5a6a',
          500: '#f42b40',
          600: '#e1102a',
          700: '#bd0a22',
          800: '#9c0c21',
          900: '#820f22',
          950: '#4d0a16',
        },
        slate: {
          850: '#172033',
          950: '#080d16',
        }
      },
      animation: {
        'pulse-fast': 'pulse 0.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slideUp 0.4s ease-out',
        'fade-in': 'fadeIn 0.3s ease-out',
        'shake': 'shake 0.5s ease-in-out',
      },
      keyframes: {
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        shake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '20%, 60%': { transform: 'translateX(-6px)' },
          '40%, 80%': { transform: 'translateX(6px)' },
        },
      },
      boxShadow: {
        'glow-red': '0 0 30px rgba(244,43,64,0.35)',
        'glow-green': '0 0 30px rgba(34,197,94,0.3)',
        'glow-yellow': '0 0 30px rgba(234,179,8,0.3)',
        'card': '0 4px 24px rgba(0,0,0,0.12)',
        'card-dark': '0 4px 24px rgba(0,0,0,0.5)',
      }
    },
  },
  plugins: [],
}
