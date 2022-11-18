CREATE TABLE public.movies (
    id integer NOT NULL,
    title text,
    original_language character(2),
    runtime character varying(10),
    revenue numeric,
    release_date character varying(30),
    production_country character varying(100),
    dir_id integer,
    vote_average real,
    vote_count integer,
    production_company text
);

COPY movies
-- Local directory path
FROM 'impact_and_comparative_summarization/data/movies.tsv'
DELIMITER '|'
CSV HEADER;


CREATE TABLE public.director (
    id integer NOT NULL,
    name text,
    gender integer
);

COPY director
-- Local directory path
FROM 'impact_and_comparative_summarization/data/directors.tsv'
DELIMITER '|'
CSV HEADER;

CREATE TABLE public.genre (
    movie_id integer NOT NULL,
    genre character varying(30)
);

COPY genre
-- Local directory path
FROM 'impact_and_comparative_summarization/data/genres.tsv'
DELIMITER '|'
CSV HEADER;
