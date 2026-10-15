// Hero background: warm late-afternoon light, a paper grain, and a faint line drawing of a road home.
// CSS and inline SVG only, purely decorative.

export function HeroBackdrop() {
  return (
    <div className="lpb" aria-hidden>
      <div className="lpb-light lpb-sun" />
      <div className="lpb-light lpb-peach" />
      <div className="lpb-light lpb-sage" />
      <svg className="lpb-home" viewBox="0 0 1440 240" preserveAspectRatio="xMidYMax slice" fill="none">
        <g className="lpb-ink" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
          {/* ground */}
          <path d="M0 214 C 260 200, 520 222, 780 210 S 1240 198, 1440 210" />
          {/* far rooftops, left */}
          <path className="lpb-far" d="M90 210 V178 L120 156 L150 178 V210 M150 192 H196 V170 L222 150 L248 170 V206 M262 204 V186 L282 170 L302 186 V203" />
          <path className="lpb-far" d="M206 184 h12 v10 h-12 z" />
          {/* far rooftops, right */}
          <path className="lpb-far" d="M1250 204 V180 L1276 160 L1302 180 V203 M1316 203 V188 L1334 174 L1352 188 V204" />
          {/* tree by the house */}
          <path d="M1012 206 V176" />
          <path d="M1012 178 C 990 176, 988 150, 1004 142 C 1006 126, 1028 124, 1032 140 C 1046 146, 1040 176, 1012 178 Z" />
          {/* the house */}
          <path d="M1062 206 V140 L1112 104 L1162 140 V205" />
          <path d="M1052 146 L1112 100 L1172 146" />
          <path d="M1138 118 V100 H1150 V127" />
          <path d="M1122 205 V168 H1142 V205" />
          {/* the path home */}
          <path className="lpb-path" d="M600 240 C 760 230, 980 222, 1132 206" />
        </g>
        {/* one warmly lit window */}
        <rect className="lpb-window" x="1078" y="150" width="26" height="22" rx="2" />
        <path className="lpb-ink" strokeWidth="1.2" d="M1091 150 V172 M1078 161 H1104" />
      </svg>
    </div>
  );
}
