// In right-to-left text, runs of Latin letters, digits and units (medicine names, doses, times) are isolated as
// left-to-right so "Apixaban 5 mg" and "10:30" never come out scrambled.
const LTR_RUN = /([A-Za-z0-9][A-Za-z0-9.,:/%°+\-]*(?:\s+[A-Za-z0-9.,:/%°+\-]+)*)/g;

export function Bidi({ text, rtl }: { text: string; rtl?: boolean }) {
  if (!rtl) return <>{text}</>;
  const parts = text.split(LTR_RUN);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 === 1 ? (
          <bdi key={i} dir="ltr">
            {p}
          </bdi>
        ) : (
          p
        ),
      )}
    </>
  );
}
